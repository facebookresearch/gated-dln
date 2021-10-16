"""Class to interface with an Experiment"""
from __future__ import annotations

import pathlib

import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import nn
from xplogger.logbook import LogBook

from src.checkpoint import utils as checkpoint_utils
from src.experiment.ds import ExperimentMetadata, TasksForKPathModel
from src.model import utils as model_utils
from src.model.base import Model as BaseModelCls

USE_MOE = True


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
        weight_init: dict,
        gate_cfg: DictConfig,
        description: str = "k path model. We train O(k) paths and evaluate on O(k**2) paths.",
    ):
        super().__init__(name=name, model_cfg=None, description=description)  # type: ignore[arg-type]
        # error: Argument "model_cfg" to "__init__" of "Model" has incompatible type "None"; expected "DictConfig"

        assert encoder_cfg["should_share"] is False

        assert decoder_cfg["should_share"] is False

        self.tasks = tasks

        hidden_size = hidden_layer_cfg["dim"]

        self.encoders = nn.ModuleList(
            [
                model_utils.get_encoder(
                    in_features=self.tasks.in_features,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    should_use_non_linearity=should_use_non_linearity,
                )
                for _ in range(self.tasks.shape[0])
            ]
        )

        if USE_MOE:
            self.decoders = model_utils.get_moe_decoder(
                num_experts=self.tasks.shape[0],
                out_features=self.tasks.out_features,
                num_layers=num_layers,
                hidden_size=hidden_size,
                should_use_non_linearity=should_use_non_linearity,
            )
        else:
            self.decoders = nn.ModuleList(
                [
                    model_utils.get_decoder(
                        out_features=self.tasks.out_features,
                        num_layers=num_layers,
                        hidden_size=hidden_size,
                        should_use_non_linearity=should_use_non_linearity,
                    )
                    for _ in range(self.tasks.shape[1])
                ]
            )

        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"], bias=weight_init["bias"]
            )
            self.encoders.apply(init_weights)
            self.decoders.apply(init_weights)
        self.encoders = torch.jit.script(self.encoders)
        self.decoders = torch.jit.script(self.decoders)

        if USE_MOE:
            self.get_decoder_output = self.get_decoder_output_using_moe
        else:
            self.get_decoder_output = self.get_decoder_output_using_model_list

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.gate_cfg = gate_cfg

        self.gate = self.make_gate()

    def make_gate(self):
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

    def _get_input_output_map(self, mode: str) -> list[tuple(int, int)]:
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
            for i in range(self.gate_cfg["num_classes_in_original_dataset"]):
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

    def load(self, name: str, save_dir: str, step: int, logbook: LogBook) -> "Model":
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
        weight_init: dict,
        gate_cfg: DictConfig,
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
            weight_init=weight_init,
            gate_cfg=gate_cfg,
            description=description,
        )  # type: ignore[arg-type]
        # error: Argument "model_cfg" to "__init__" of "Model" has incompatible type "None"; expected "DictConfig"

        hidden_size = hidden_layer_cfg["dim"]

        self.hidden_layer = model_utils.get_hidden(
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"], bias=weight_init["bias"]
            )
            self.hidden_layer.apply(init_weights)

        self.hidden_layer = torch.jit.script(self.hidden_layer)

        assert hidden_layer_cfg["should_share"] is True

    def get_decoder_output_using_moe(
        self, hidden: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        outputs = self.decoders(hidden)
        outputs = (
            outputs.permute(1, 0, 2)
            # .view(
            #     batch_size,
            #     self.tasks.shape[0],
            #     self.tasks.shape[1],
            #     self.tasks.out_features,
            # )
            .reshape(-1, self.tasks.out_features)
        )
        return outputs

    def get_decoder_output_using_model_list(
        self, hidden: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        outputs = [
            decoder(hidden)
            .reshape(batch_size, self.tasks.shape[0], self.tasks.out_features)
            .unsqueeze(2)
            for decoder in self.decoders
        ]  # list of size self.tasks.shape[1]
        # outputs[0].shape == (batch, self.tasks.shape[0], 1, 2)

        output_tensor = torch.cat(outputs, dim=2).view(-1, self.tasks.out_features)
        return output_tensor

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # (batch_size * self.tasks.shape[0], dim)

        output_tensor = self.get_decoder_output(hidden=hidden, batch_size=batch_size)

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
