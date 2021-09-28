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

        self.hidden_layer = model_utils.get_hidden(
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
        )
        if weight_init["should_do"]:
            init_weights = model_utils.get_weight_init_fn(
                gain=weight_init["gain"], bias=weight_init["bias"]
            )
            self.encoders.apply(init_weights)
            self.decoders.apply(init_weights)
            self.hidden_layer.apply(init_weights)

        self.encoders = torch.jit.script(self.encoders)
        self.decoders = torch.jit.script(self.decoders)
        self.hidden_layer = torch.jit.script(self.hidden_layer)

        assert encoder_cfg["should_share"] is False

        assert hidden_layer_cfg["should_share"] is True

        assert decoder_cfg["should_share"] is False

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self.gate_cfg = gate_cfg

        self.gate = self.make_gate()

    def make_gate(self):
        if self.gate_cfg["mode"] == "fully_connected":
            gate = torch.ones(*self.tasks.shape, device="cuda", dtype=torch.float32)
            return gate

        input_output_map = []
        if "_plus_mod" in self.gate_cfg["mode"]:
            num_cols = int(self.gate_cfg["mode"].split("_plus_mod")[0])
            num_rows = self.tasks.shape[0]
            for i in range(num_rows):
                for j in range(num_cols):
                    input_output_map.append((i, (i + j) % num_rows))
        elif "_plus_minus_mod" in self.gate_cfg["mode"]:
            num_cols = int(self.gate_cfg["mode"].split("_plus_minus_mod")[0])
            num_rows = self.tasks.shape[0]
            for i in range(num_rows):
                for j in range(-1 * ((num_cols - 1) // 2), (num_cols // 2) + 1):
                    input_output_map.append((i, (i + j + num_rows) % num_rows))
        elif self.gate_cfg["mode"] == "mod":
            for i in range(self.gate_cfg["num_classes_in_original_dataset"]):
                input_output_map.append((i, i))
        else:
            raise NotImplementedError(f"mode = self.gate_cfg['mode'] is not supported.")
        gate = torch.zeros(*self.tasks.shape, device="cuda", dtype=torch.float32)
        for current_input, current_output in input_output_map:
            gate[current_input][current_output] = 1.0
        print(gate)
        print(gate.shape, gate.sum().item())
        return gate

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, metadata: ExperimentMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.shape[0]

        transformed_x = [transform(x) for transform in self.tasks.input_transforms]

        features = [encoder(x) for encoder, x in zip(self.encoders, transformed_x)]

        hidden = self.hidden_layer(
            torch.cat(features, dim=1).view(-1, features[0].shape[1])
        )
        # (batch_size * self.tasks.shape[0], dim)

        outputs = [
            decoder(hidden).reshape(batch_size, self.tasks.shape[0], 2).unsqueeze(2)
            for decoder in self.decoders
        ]  # list of size self.tasks.shape[1]
        # outputs[0].shape == (batch, self.tasks.shape[0], 1, 2)
        output_tensor = torch.cat(outputs, dim=2).view(-1, 2)
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
        # breakpoint()
        # x = target_tensor.cpu().view(
        #     batch_size, self.tasks.shape[0], self.tasks.shape[1]
        # )[:, 0, 0]
        # y = (
        #     output_tensor.max(dim=1)[1]
        #     .cpu()
        #     .view(batch_size, self.tasks.shape[0], self.tasks.shape[1])[:, 0, 0]
        # )

        # if metadata.mode == ExperimentMode("test"):
        #     print(
        #         f"f1_score: {f1_score(x, y)}, precision: {precision_score(x, y)}, recall: {recall_score(x, y)}, accuracy: {accuracy_score(x, y)}"
        #     )

        return loss.detach(), loss_to_backprop, num_correct
