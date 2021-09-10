"""Class to interface with an Experiment"""
from __future__ import annotations

import torch
import torch.utils.data
from torch import nn

from src.data.batch_list import BatchList
from src.experiment.ds import ExperimentMetadata, Task
from src.model import utils as model_utils
from src.model.base import Model as BaseModel


class Model(BaseModel):
    def __init__(
        self,
        name: str,
        task_one: Task,
        task_two: Task,
        in_features: int,
        out_features: int,
        num_layers: int,
        encoder_cfg: dict,
        hidden_layer_cfg: dict,
        decoder_cfg: dict,
        should_use_non_linearity: bool,
        weight_init: dict,
        description: str = "Four path model. We train three paths and evaluate on the fourth path.",
    ):
        super().__init__(name=name, model_cfg=None, description=description)  # type: ignore[arg-type]
        # error: Argument "model_cfg" to "__init__" of "Model" has incompatible type "None"; expected "DictConfig"

        self.task_one_name = task_one.name

        self.task_two_name = task_two.name
        self.should_use_two_batches = True

        if self.should_use_two_batches:

            def new_task_one_transform(x: BatchList):
                return task_one.transform(x[0])

            def new_task_two_transform(x: BatchList):
                return task_two.transform(x[1])

            # def new_task_one_target_transform(x: BatchList):
            #     return task_one.target_transform(x[0])

            # def new_task_two_target_transform(x: BatchList):
            #     return task_two.target_transform(x[1])

            self.task_one_transform = new_task_one_transform
            self.task_two_transform = new_task_two_transform
            # self.task_one_target_transform = new_task_one_target_transform
            # self.task_two_target_transform = new_task_two_target_transform

        else:
            self.task_one_transform = task_one.transform
            self.task_two_transform = task_two.transform
        self.task_one_target_transform = task_one.target_transform
        self.task_two_target_transform = task_two.target_transform

        hidden_size = hidden_layer_cfg["dim"]

        task_one_encoder = model_utils.get_encoder(
            in_features=in_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        task_one_hidden_layers = model_utils.get_hidden(
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        task_one_decoder = model_utils.get_decoder(
            out_features=out_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )

        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"], bias=weight_init["bias"]
            )
            task_one_encoder.apply(init_weights)
            task_one_hidden_layers.apply(init_weights)
            task_one_decoder.apply(init_weights)

        if encoder_cfg["should_share"]:
            task_two_encoder = task_one_encoder
        else:
            task_two_encoder = model_utils.get_encoder(
                in_features=in_features,
                num_layers=num_layers,
                hidden_size=hidden_size,
                should_use_non_linearity=should_use_non_linearity,
            )
        self.should_share_hidden_layer = hidden_layer_cfg["should_share"]
        if self.should_share_hidden_layer:
            task_two_hidden_layers = task_one_hidden_layers
        else:
            task_two_hidden_layers = model_utils.get_hidden(
                num_layers=num_layers,
                hidden_size=hidden_size,
                should_use_non_linearity=should_use_non_linearity,
            )

        if decoder_cfg["should_share"]:
            task_two_decoder = task_one_decoder
        else:
            task_two_decoder = model_utils.get_decoder(
                out_features=out_features,
                num_layers=num_layers,
                hidden_size=hidden_size,
                should_use_non_linearity=should_use_non_linearity,
            )

        self.task_one_model = model_utils.get_container_model(
            model_list=[task_one_encoder, task_one_hidden_layers, task_one_decoder],
            should_use_non_linearity=should_use_non_linearity,
        )

        self.task_two_model = model_utils.get_container_model(
            model_list=[task_two_encoder, task_two_hidden_layers, task_two_decoder],
            should_use_non_linearity=should_use_non_linearity,
        )

        if self.should_share_hidden_layer:
            self.task_one_encoder_task_two_decoder = model_utils.get_container_model(
                model_list=[task_one_encoder, task_one_hidden_layers, task_two_decoder],
                should_use_non_linearity=should_use_non_linearity,
            )
            self.task_two_encoder_task_one_decoder = model_utils.get_container_model(
                model_list=[task_two_encoder, task_one_hidden_layers, task_one_decoder],
                should_use_non_linearity=should_use_non_linearity,
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
        output = {
            "task_one": self.task_one_model(self.task_one_transform(x)),
            "task_two": self.task_two_model(self.task_two_transform(x)),
        }
        if self.should_use_two_batches:
            target = {
                "task_one": self.task_one_target_transform(y[0]),
                "task_two": self.task_two_target_transform(y[1]),
            }
        else:
            target = {
                "task_one": self.task_one_target_transform(y),
                "task_two": self.task_two_target_transform(y),
            }
        if self.should_share_hidden_layer:
            key = "task_two_encoder_task_one_decoder"
            output[key] = self.task_two_encoder_task_one_decoder(
                self.task_two_transform(x)
            )
            if self.should_use_two_batches:
                target[key] = self.task_one_target_transform(y[1])
            else:
                target[key] = target["task_one"]

            key = "task_one_encoder_task_two_decoder"
            output[key] = self.task_one_encoder_task_two_decoder(
                self.task_one_transform(x)
            )
            if self.should_use_two_batches:
                target[key] = self.task_two_target_transform(y[0])
            else:
                target[key] = target["task_two"]

        loss = {key: self.loss_fn(output[key], target[key]) for key in output}
        for key in loss:
            if (
                key.startswith("train_on_")
                or key == "task_two_encoder_task_one_decoder"
            ):
                loss[key] = loss[key].detach()
            elif key == "task_one_encoder_task_two_decoder":
                loss[key] = loss[key]
        num_correct = {
            key: output[key].max(1)[1].eq(target[key]).sum().item() for key in output
        }
        return output, loss, num_correct
