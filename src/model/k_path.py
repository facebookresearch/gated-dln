# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Class to interface with an Experiment"""
from __future__ import annotations

import pathlib

import hydra
import torch
import torch.utils.data
from functorch import vmap
from moe.data.feature import Feature
from omegaconf import DictConfig
from torch import nn
from xplogger.logbook import LogBook

from src.checkpoint import utils as checkpoint_utils
from src.ds.experiment import ExperimentMetadata
from src.ds.task import TasksForKPathModel
from src.model import utils as model_utils
from src.model.base import Model as BaseModelCls


class BaseModel(BaseModelCls):
    def __init__(
        self,
        name: str,
        tasks: TasksForKPathModel,
        num_layers: int,
        encoder_cfg: dict,
        hidden_layer_cfg: dict,
        decoder_cfg: dict,
        should_use_non_linearity: bool,
        non_linearity_cfg: DictConfig,
        weight_init: dict,
        gate_cfg: DictConfig,
        pretrained_cfg: DictConfig,
        should_use_preprocessed_dataset: bool,
        description: str = "k path model. We train O(k) paths and evaluate on O(k**2) paths.",
    ):
        super().__init__(name=name, model_cfg=None, description=description)  # type: ignore[arg-type]
        # error: Argument "model_cfg" to "__init__" of "Model" has incompatible type "None"; expected "DictConfig"

        assert encoder_cfg["should_share"] is False

        assert decoder_cfg["should_share"] is False
        (
            self._pretrained_model,
            in_features,
            self.should_use_pretrained_model,
        ) = hydra.utils.instantiate(pretrained_cfg)

        if pretrained_cfg["should_finetune"]:
            self.get_output_from_pretrained_model = (
                self.get_output_from_pretrained_model_in_train_mode
            )

        else:
            self.get_output_from_pretrained_model = (
                self.get_output_from_pretrained_model_in_inference_mode
            )

        self.get_output_from_pretrained_model = vmap(
            self.get_output_from_pretrained_model
        )

        self.tasks = tasks

        if in_features == -1:
            in_features = self.tasks.in_features

        self.should_use_preprocessed_dataset = should_use_preprocessed_dataset

        hidden_size = hidden_layer_cfg["dim"]
        self.encoders = nn.ModuleList(
            [
                model_utils.get_encoder(
                    in_features=in_features,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    should_use_non_linearity=should_use_non_linearity,
                    non_linearity_cfg=non_linearity_cfg,
                    should_use_pretrained_features=self.should_use_preprocessed_dataset,
                )
                for _ in range(self.tasks.shape[0])
            ]
        )

        print(self.encoders)
        self.decoders = model_utils.get_moe_decoder(
            num_experts=self.tasks.shape[0],
            out_features=self.tasks.out_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
            non_linearity_cfg=non_linearity_cfg,
        )
        print(self.decoders)
        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"],
                bias=weight_init["bias"],
                name=weight_init.get("name", "xavier_uniform_"),
            )
            self.encoders.apply(init_weights)
            self.decoders.apply(init_weights)
        _batch_size = 8
        dummy_inputs = {
            "encoders": torch.ones(_batch_size, 784),
            "decoders": torch.ones(_batch_size, hidden_size),
        }

        self.get_decoder_output = self.get_decoder_output_using_moe

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.gate_cfg = gate_cfg

        self.gate: torch.Tensor = self.make_gate()

    @torch.no_grad()
    def get_output_from_pretrained_model_in_no_grad_mode(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self._pretrained_model(x)

    @torch.inference_mode()
    def get_output_from_pretrained_model_in_inference_mode(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self._pretrained_model(x)

    def get_output_from_pretrained_model_in_train_mode(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self._pretrained_model(x)

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if self.should_use_pretrained_model:
            self._pretrained_model.eval()
        # we want the pretrained model to be in eval mode all the time.
        return self

    def make_gate(self) -> torch.Tensor:
        if self.gate_cfg["mode"] == "fully_connected":
            # this value should come from the gate_cfg
            gate = torch.ones(*self.tasks.shape, device="cpu", dtype=torch.float32)
            return gate

        input_output_map = self._get_input_output_map(mode=self.gate_cfg["mode"])
        gate = torch.zeros(*self.tasks.shape, device="cpu", dtype=torch.float32)
        for current_input, current_output in input_output_map:
            gate[current_input][current_output] = 1.0
        if self.gate_cfg["mode"].endswith("permute"):
            gate = gate[:, torch.randperm(gate.shape[1])]
        print(gate)
        print(gate.shape, gate.sum().item())
        return gate

    def _get_input_output_map(self, mode: str) -> list[tuple[int, int]]:
        input_output_map = []
        if "_plus_mod" in mode:
            num_cols = int(mode.split("_plus_mod")[0])
            num_rows = self.tasks.shape[0]
            for i in range(num_rows):
                for j in range(num_cols):
                    input_output_map.append((i, (i + j) % num_rows))
        elif "_plus_minus_mod" in mode:
            num_cols = int(mode.split("_plus_minus_mod")[0])
            num_rows = self.tasks.shape[0]
            for i in range(num_rows):
                for j in range(-1 * ((num_cols - 1) // 2), (num_cols // 2) + 1):
                    input_output_map.append((i, (i + j + num_rows) % num_rows))
        elif mode == "mod":
            for i in range(self.gate_cfg["num_classes_in_selected_dataset"]):
                input_output_map.append((i, i))
        else:
            raise NotImplementedError(f"mode = {mode} is not supported.")
        return input_output_map

    def to(self, device, *args, **kwargs):
        self.gate = self.gate.to(device)
        return super().to(device, *args, **kwargs)

    def save(
        self,
        name: str,
        save_dir_path: pathlib.Path,
        step: int,
        retain_last_n: int,
        logbook: LogBook,
    ) -> None:
        checkpoint_utils.save_gate(
            gate=self.gate, save_dir_path=save_dir_path, logbook=logbook
        )
        return super().save(
            name=name,
            save_dir_path=save_dir_path,
            step=step,
            retain_last_n=retain_last_n,
            logbook=logbook,
        )

    def load(self, name: str, save_dir: str, step: int, logbook: LogBook) -> nn.Module:
        self.gate = checkpoint_utils.load_gate(save_dir=save_dir, logbook=logbook)
        return super().load(name=name, save_dir=save_dir, step=step, logbook=logbook)
        # mpyp error: Incompatible return value type (got Module, expected "Model")  [return-value]


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
        non_linearity_cfg: DictConfig,
        weight_init: dict,
        gate_cfg: DictConfig,
        pretrained_cfg: DictConfig,
        should_use_preprocessed_dataset: bool,
        description: str = "k path model. We train O(k) paths and evaluate on O(k**2) paths.",
    ):
        super().__init__(
            name=name,
            tasks=tasks,
            num_layers=num_layers,
            encoder_cfg=encoder_cfg,
            hidden_layer_cfg=hidden_layer_cfg,
            decoder_cfg=decoder_cfg,
            should_use_non_linearity=should_use_non_linearity,
            non_linearity_cfg=non_linearity_cfg,
            weight_init=weight_init,
            gate_cfg=gate_cfg,
            pretrained_cfg=pretrained_cfg,
            should_use_preprocessed_dataset=should_use_preprocessed_dataset,
            description=description,
        )

        hidden_size = hidden_layer_cfg["dim"]

        self.hidden_layer = model_utils.get_hidden(
            num_layers=hidden_layer_cfg["num_layers"],
            hidden_size=hidden_size,
            should_use_non_linearity=hidden_layer_cfg["should_use_non_linearity"],
            non_linearity_cfg=hidden_layer_cfg["non_linearity_cfg"],
            recurrence_cfg=hidden_layer_cfg["recurrence_cfg"],
        )

        # self.hidden_layer = model_utils.get_hidden(
        #     num_layers=hidden_layer_cfg.get("num_layers", num_layers),
        #     hidden_size=hidden_size,
        #     should_use_non_linearity=hidden_layer_cfg.get(
        #         "should_use_non_linearity", should_use_non_linearity
        #     ),
        #     non_linearity_cfg=hidden_layer_cfg.get(
        #         "non_linearity_cfg", non_linearity_cfg
        #     ),
        # )

        print(self.hidden_layer)

        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"],
                bias=weight_init["bias"],
                name=weight_init.get("name", "xavier_uniform_"),
            )
            self.hidden_layer.apply(init_weights)

        _batch_size = 8
        dummy_inputs = {
            "hidden_layer": torch.ones(_batch_size, hidden_size),
        }
        self.hidden_layer = torch.jit.trace(
            self.hidden_layer, (dummy_inputs["hidden_layer"])
        )

        assert hidden_layer_cfg["should_share"] is True

        if self.should_use_preprocessed_dataset:
            self.forward = self.forward_when_using_preprocessed_dataset  # type: ignore[assignment]
            # error: Cannot assign to a method
            self.forward_eval = self.forward_when_using_preprocessed_dataset_functorch  # type: ignore[assignment]
            # error: Cannot assign to a method
        else:
            self.forward = self.forward_when_using_unprocessed_dataset  # type: ignore[assignment]
            # error: Cannot assign to a method
            self.forward_eval = self.forward_when_using_unprocessed_dataset_functorch

    def get_decoder_output_using_moe(
        self, hidden: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        outputs = self.decoders(hidden)
        outputs = outputs.permute(1, 0, 2).reshape(-1, self.tasks.out_features)
        return outputs

    def common_forward_when_using_unprocessed_dataset(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        hidden = self.hidden_layer(
            features.reshape(batch_size * self.tasks.shape[0], features.shape[2])
        )
        # (batch_size * self.tasks.shape[0], dim)

        output_tensor = self.get_decoder_output(hidden=hidden, batch_size=batch_size)  # type: ignore[operator]
        # error: "Tensor" not callable  [operator]
        # (batch, self.tasks.shape[0], self.tasks.shape[1], 2)
        transformed_y = [transform(y) for transform in self.tasks.target_transforms]
        # list of size self.tasks.shape[1]
        # transformed_y[0].shape == (batch,)

        target_tensor = torch.cat(
            [
                current_y.unsqueeze(1).repeat(1, self.tasks.shape[0]).unsqueeze(2)
                for current_y in transformed_y
            ],
            dim=2,
        ).view(-1)

        loss = self.loss_fn(input=output_tensor, target=target_tensor)

        loss = loss.view(batch_size, self.tasks.shape[0], self.tasks.shape[1]).mean(
            dim=0
        )

        gated_loss = loss * self.gate

        loss_to_backprop = gated_loss.sum() / self.gate.sum()
        num_correct = (
            output_tensor.max(dim=1)[1]
            .eq(target_tensor)
            .view(batch_size, self.tasks.shape[0], self.tasks.shape[1])
            .sum(dim=0)
        )
        return loss.detach(), loss_to_backprop, num_correct

    def forward_when_using_unprocessed_dataset(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed_x = self.get_output_from_pretrained_model(
            torch.cat(
                [
                    transform(x).unsqueeze(0).clone()
                    for transform in self.tasks.input_transforms
                ],
                dim=0,
            )
        )

        features: list[Feature] = [
            encoder(x).unsqueeze(1) for encoder, x in zip(self.encoders, transformed_x)
        ]

        features = torch.cat(features, dim=1)

        return self.common_forward_when_using_unprocessed_dataset(
            x=x, features=features, y=y
        )

    def forward_when_using_unprocessed_dataset_functorch(
        self,
        encoder_fmodel,
        encoder_params,
        encoder_buffers,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed_x = self.get_output_from_pretrained_model(
            torch.cat(
                [
                    transform(x).unsqueeze(0).clone()
                    for transform in self.tasks.input_transforms
                ],
                dim=0,
            )
        )

        features: Feature = vmap(encoder_fmodel)(
            encoder_params,
            encoder_buffers,
            transformed_x,
        ).permute(1, 0, 2)

        return self.common_forward_when_using_unprocessed_dataset(
            x=x, features=features, y=y
        )

    def common_forward_when_using_preprocessed_dataset(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        hidden = self.hidden_layer(
            features.reshape(batch_size * self.tasks.shape[0], features.shape[2])
        )
        # (batch_size * self.tasks.shape[0], dim)
        output_tensor = self.get_decoder_output(hidden=hidden, batch_size=batch_size)  # type: ignore[operator]
        # error: "Tensor" not callable  [operator]
        # (batch, self.tasks.shape[0], self.tasks.shape[1], 2)
        target_tensor = y.view(-1)

        loss = self.loss_fn(input=output_tensor, target=target_tensor)

        loss = loss.view(batch_size, self.tasks.shape[0], self.tasks.shape[1]).mean(
            dim=0
        )

        gated_loss = loss * self.gate

        loss_to_backprop = gated_loss.sum() / self.gate.sum()
        num_correct = (
            output_tensor.max(dim=1)[1]
            .eq(target_tensor)
            .view(batch_size, self.tasks.shape[0], self.tasks.shape[1])
            .sum(dim=0)
        )
        return loss.detach(), loss_to_backprop, num_correct

    def forward_when_using_preprocessed_dataset_functorch(
        self,
        encoder_fmodel,
        encoder_params,
        encoder_buffers,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed_x = x.permute(1, 0, 2)
        features: Feature = vmap(encoder_fmodel)(
            encoder_params,
            encoder_buffers,
            transformed_x,
        ).permute(1, 0, 2)
        return self.common_forward_when_using_preprocessed_dataset(
            features=features, x=x, y=y
        )

    def forward_when_using_preprocessed_dataset(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = [
            encoder(x).unsqueeze(1)
            for encoder, x in zip(self.encoders, x.permute(1, 0, 2))
        ]
        features = torch.cat(features, dim=1)
        return self.common_forward_when_using_preprocessed_dataset(
            features=features, x=x, y=y
        )

    def extract_features_for_caching_dataset(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        transformed_x = (
            self.get_output_from_pretrained_model(
                torch.cat(
                    [transform(x) for transform in self.tasks.input_transforms], dim=0
                )
            )
            .view(len(self.tasks.input_transforms), batch_size, -1)
            .permute(1, 0, 2)
        )
        transformed_y_list = [
            transform(y).unsqueeze(1).repeat(1, self.tasks.shape[0]).unsqueeze(2)
            for transform in self.tasks.target_transforms
        ]
        transformed_y = torch.cat(transformed_y_list, dim=2)
        return transformed_x, transformed_y  # type: ignore[return-value]
        # Incompatible return value type (got "Tuple[Tensor, List[Any]]", expected "Tuple[Tensor, Tensor, Tensor]")

    def extract_features(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch_size = x.shape[0]

        transformed_x = [transform(x) for transform in self.tasks.input_transforms]
        features = [
            encoder(x).unsqueeze(1) for encoder, x in zip(self.encoders, transformed_x)
        ]
        hidden = self.hidden_layer(
            torch.cat(features, dim=1).view(
                batch_size * self.tasks.shape[0], features[0].shape[2]
            )
        )
        return features, hidden
