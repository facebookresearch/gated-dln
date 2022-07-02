"""Class to interface with an Experiment"""
from __future__ import annotations

from typing import Any

import hydra
import torch
import torch.utils.data
from moe.model import layer as moe_layer
from omegaconf.dictconfig import DictConfig
from torch import nn

from src.model.moe import model as moe_model


def get_weight_init_fn(gain: float, bias: float = 0.01, name: str = "xavier_uniform_"):
    def init_weights(m):
        if isinstance(m, (nn.Linear, moe_layer.Linear)):
            if name == "xavier_uniform_":
                torch.nn.init.xavier_uniform_(
                    m.weight,
                    gain=gain,
                )
            elif name == "xavier_normal_":
                torch.nn.init.xavier_normal_(
                    m.weight,
                    gain=gain,
                )
            else:
                raise NotImplementedError(f"name={name} is not supported.")
            m.bias.data.fill_(bias)
        elif isinstance(
            m,
            (
                nn.Flatten,
                nn.ReLU,
                nn.LeakyReLU,
                nn.Sequential,
                nn.ModuleList,
                moe_model.FeedForward,
            ),
        ):
            pass
        else:
            raise NotImplementedError(f"module = {m} is not supported.")

    return init_weights


def get_encoder(
    num_layers: int,
    in_features: int,
    hidden_size: int,
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
    should_use_pretrained_features: bool,
):
    layers: list[nn.Module]
    if should_use_pretrained_features:
        layers = []
    else:
        layers = [
            nn.Flatten(),
        ]

    layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
    for _ in range(num_layers - 1):
        if should_use_non_linearity:
            layers.append(hydra.utils.instantiate(non_linearity_cfg))
        layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
    return nn.Sequential(*layers)


def get_moe_encoder(
    num_experts: int,
    num_layers: int,
    in_features: int,
    hidden_size: int,
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
):
    layers = [
        nn.Flatten(start_dim=2, end_dim=-1),
        moe_model.FeedForward(
            num_experts=num_experts,
            in_features=in_features,
            out_features=hidden_size,
            num_layers=num_layers,
            should_use_non_linearity=should_use_non_linearity,
            non_linearity_cfg=non_linearity_cfg,
            hidden_features=hidden_size,
        ),
    ]
    return nn.Sequential(*layers)


def get_hidden(
    num_layers: int,
    hidden_size: int,
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
    recurrence_cfg: DictConfig,
):
    def _get_one_hidden_block(
        num_layers: int,
        hidden_size: int,
        should_use_non_linearity: bool,
        non_linearity_cfg: DictConfig,
    ):
        layers: list[Any] = [
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        ]
        for _ in range(num_layers - 1):
            if should_use_non_linearity:
                layers.append(hydra.utils.instantiate(non_linearity_cfg))
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        return nn.Sequential(*layers)

    if recurrence_cfg["should_use"]:
        if recurrence_cfg["should_reuse_weights"]:
            one_block = _get_one_hidden_block(
                num_layers=num_layers,
                hidden_size=hidden_size,
                should_use_non_linearity=should_use_non_linearity,
                non_linearity_cfg=non_linearity_cfg,
            )
            blocks = [one_block for _ in range(recurrence_cfg["num_blocks"])]
        else:
            blocks = [
                _get_one_hidden_block(
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    should_use_non_linearity=should_use_non_linearity,
                    non_linearity_cfg=non_linearity_cfg,
                )
                for _ in range(recurrence_cfg["num_blocks"])
            ]
        if recurrence_cfg["should_use_non_linearity"]:
            blocks_to_use = [blocks[0]]
            for block_idx in range(recurrence_cfg["num_blocks"] - 1):
                blocks_to_use.append(
                    hydra.utils.instantiate(recurrence_cfg["non_linearity_cfg"])
                )
                blocks_to_use.append(blocks[block_idx + 1])
        else:
            blocks_to_use = blocks

        return nn.Sequential(*blocks_to_use)
    else:
        return _get_one_hidden_block(
            num_layers=num_layers,
            hidden_size=hidden_size,
            should_use_non_linearity=should_use_non_linearity,
            non_linearity_cfg=non_linearity_cfg,
        )


def get_decoder(
    num_layers: int,
    out_features: int,
    hidden_size: int,
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
):
    layers: nn.ModuleList = []  # type: ignore[assignment]
    # Incompatible types in assignment (expression has type "List[<nothing>]", variable has type "ModuleList")
    for _ in range(num_layers - 1):
        if should_use_non_linearity:
            layers.append(hydra.utils.instantiate(non_linearity_cfg))
        layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
    layers += [
        nn.Linear(in_features=hidden_size, out_features=out_features),
    ]

    return nn.Sequential(*layers)


def get_moe_decoder(
    num_experts: int,
    num_layers: int,
    out_features: int,
    hidden_size: int,
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
):
    return moe_model.FeedForward(
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=out_features,
        num_layers=num_layers,
        should_use_non_linearity=should_use_non_linearity,
        non_linearity_cfg=non_linearity_cfg,
        hidden_features=hidden_size,
    )


def get_container_model(
    model_list: list[nn.Module],
    should_use_non_linearity: bool,
    non_linearity_cfg: DictConfig,
):
    container = []
    for model in model_list:
        container.append(model)
        if should_use_non_linearity:
            container.append(hydra.utils.instantiate(non_linearity_cfg))
    if should_use_non_linearity:
        container.pop()

    return nn.Sequential(*container)


def get_pretrained_model(
    should_use: bool,
    model_cfg: DictConfig,
    should_load_weights: bool,
    path_to_load_weights: str,
    should_finetune: bool,
    should_enable_jit: bool,
) -> tuple[Any, int, bool]:
    if should_use:
        model = hydra.utils.instantiate(model_cfg)
        if should_load_weights:
            checkpoint = torch.load(path_to_load_weights)
            model.load_state_dict(checkpoint)

        if should_enable_jit:
            raise ValueError(
                f"should_enable_jit={should_enable_jit}. It should not be enabled."
            )
        else:
            output_dim = model.output_dim
        assert should_finetune is False
        model = model.eval()
        model.requires_grad_(should_finetune)
        return model, output_dim, True
    else:
        return lambda x: x, -1, False
