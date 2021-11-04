from typing import List, Tuple

import torch
from torch import nn

from src.model import base as base_model
from src.utils.config import DictConfig


class Model(base_model.Model):
    """Feedforward Model."""

    def __init__(
        self,
        name: str,
        model_cfg: DictConfig,
        should_use_non_linearity: bool,
        description: str = "",
    ):
        super().__init__(name=name, model_cfg=model_cfg, description=description)
        self.model = build_model(
            model_cfg=self.model_cfg, should_use_non_linearity=should_use_non_linearity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(model_cfg: DictConfig, should_use_non_linearity: bool) -> nn.Module:
    """Build the model

    Args:
        name (str): [description]
        model_cfg (DictConfig): [description]

    Returns:
        nn.Module: [description]
    """
    input_size, output_size = get_input_output_sizes(
        dataset_name=model_cfg.dataset_name
    )
    layer_list: List[nn.Module] = [
        nn.Flatten(),
        nn.Linear(input_size, model_cfg.hidden_size),
    ]
    for _ in range(model_cfg.num_layers - 1):
        if should_use_non_linearity:
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(model_cfg.hidden_size, model_cfg.hidden_size))
    if should_use_non_linearity:
        layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(model_cfg.hidden_size, output_size))

    return nn.Sequential(*layer_list)


def get_input_output_sizes(dataset_name: str) -> Tuple[int, int]:
    if dataset_name == "mnist":
        return (784, 10)
    elif dataset_name == "cifar10":
        return (512, 10)
    raise ValueError(f"dataset_name={dataset_name} is not supported")
