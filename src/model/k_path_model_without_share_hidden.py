"""Class to interface with an Experiment"""
from __future__ import annotations

import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import nn

from src.experiment.ds import ExperimentMetadata, TasksForKPathModel
from src.model import utils as model_utils
from src.model.base import Model as BaseModel


class Model(BaseModel):
    def __init__(
        self,
        name: str,
        tasks: TasksForKPathModel,
        num_layers: int,
        encoder_cfg: dict,
        hidden_layer_cfg: dict,
        decoder_cfg: dict,
        should_use_non_linearity: bool,
        weight_init: dict,
        gate_cfg: DictConfig,
        description: str = "k path model. We train O(k) paths and evaluate on O(k**2) paths.",
    ):
        super().__init__(name=name, model_cfg=None, description=description)  # type: ignore[arg-type]
        # error: Argument "model_cfg" to "__init__" of "Model" has incompatible type "None"; expected "DictConfig"

        self.tasks = tasks

        assert self.tasks.shape[0] == self.tasks.shape[1]
        assert encoder_cfg["should_share"] is False

        assert hidden_layer_cfg["should_share"] is False

        assert decoder_cfg["should_share"] is False

        self.models = nn.ModuleList(
            [
                self._make_model(
                    num_layers=num_layers,
                    hidden_layer_cfg=hidden_layer_cfg,
                    should_use_non_linearity=should_use_non_linearity,
                )
                for _ in range(self.tasks.shape[0])
            ]
        )

        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"], bias=weight_init["bias"]
            )
            self.models.apply(init_weights)

        self.models = torch.jit.script(self.models)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _make_model(
        self, num_layers: int, hidden_layer_cfg: dict, should_use_non_linearity: bool
    ) -> nn.Module:
        hidden_size = hidden_layer_cfg["dim"]
        encoder = model_utils.get_encoder(
            in_features=self.tasks.in_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        hidden = model_utils.get_hidden(
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        decoder = model_utils.get_decoder(
            out_features=self.tasks.out_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        return nn.Sequential(encoder, hidden, decoder)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        outputs = [
            model(transform(x)).unsqueeze(1)
            for model, transform in zip(self.models, self.tasks.input_transforms)
        ]
        output_tensor = torch.cat(outputs, dim=1).view(-1, 2)

        transformed_y = [
            transform(y).unsqueeze(1) for transform in self.tasks.target_transforms
        ]
        target_tensor = torch.cat(transformed_y, dim=1).view(-1)
        loss = self.loss_fn(input=output_tensor, target=target_tensor).mean()
        loss_to_backprop = loss
        num_correct = output_tensor.max(dim=1)[1].eq(target_tensor).sum(dim=0) / len(
            self.tasks.input_transforms
        )
        return loss.detach(), loss_to_backprop, num_correct
