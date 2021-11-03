"""Class to interface with an Experiment"""
from __future__ import annotations

from typing import Any

import torch
import torch.utils.data
from torch import nn

from src.model.moe import layer as moe_layer


def get_weight_init_fn(gain: float, bias: float = 0.01):
    def init_weights(m):
        if isinstance(m, (nn.Linear, moe_layer.Linear)):
            torch.nn.init.xavier_uniform_(
                m.weight,
                gain=gain,
            )
            m.bias.data.fill_(bias)
        elif isinstance(
            m,
            (nn.Flatten, nn.ReLU, nn.Sequential, nn.ModuleList, moe_layer.FeedForward),
        ):
            pass
        else:
            raise NotImplementedError(f"module = {m} is not supported.")

    return init_weights


def get_encoder(
    num_layers: int, in_features: int, hidden_size: int, should_use_non_linearity: bool
):
    layers = [
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=hidden_size),
    ]
    for _ in range(num_layers - 1):
        if should_use_non_linearity:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
    return nn.Sequential(*layers)


def get_moe_encoder(
    num_experts: int,
    num_layers: int,
    in_features: int,
    hidden_size: int,
    should_use_non_linearity: bool,
):
    layers = [
        nn.Flatten(start_dim=2, end_dim=-1),
        moe_layer.FeedForward(
            num_experts=num_experts,
            in_features=in_features,
            out_features=hidden_size,
            num_layers=num_layers,
            should_use_non_linearity=should_use_non_linearity,
            hidden_features=hidden_size,
        ),
    ]
    return nn.Sequential(*layers)


def get_hidden(num_layers: int, hidden_size: int, should_use_non_linearity: bool):
    layers: list[Any] = [
        nn.Linear(in_features=hidden_size, out_features=hidden_size),
    ]
    for _ in range(num_layers - 1):
        if should_use_non_linearity:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
    return nn.Sequential(*layers)


def get_decoder(
    num_layers: int, out_features: int, hidden_size: int, should_use_non_linearity: bool
):
    layers: nn.ModuleList = []  # type: ignore[assignment]
    # Incompatible types in assignment (expression has type "List[<nothing>]", variable has type "ModuleList")
    for _ in range(num_layers - 1):
        if should_use_non_linearity:
            layers.append(nn.ReLU())
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
):
    return moe_layer.FeedForward(
        num_experts=num_experts,
        in_features=hidden_size,
        out_features=out_features,
        num_layers=num_layers,
        should_use_non_linearity=should_use_non_linearity,
        hidden_features=hidden_size,
    )


def get_container_model(model_list: list[nn.Module], should_use_non_linearity: bool):
    container = []
    for model in model_list:
        container.append(model)
        if should_use_non_linearity:
            container.append(nn.ReLU())
    if should_use_non_linearity:
        container.pop()

    return nn.Sequential(*container)
