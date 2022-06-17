from __future__ import annotations

from typing import Callable

import torch
from attrs import define


@define
class Task:
    name: str
    index_for_input_transform: int
    index_for_output_transform: int
    transform: Callable[[torch.Tensor], torch.Tensor]
    target_transform: Callable[[torch.Tensor], torch.Tensor]


@define
class TasksForKPathModel:
    tasks: list[Task]
    in_features: int
    out_features: int
    shape: tuple[int, int]
    input_transforms: list
    target_transforms: list


@define
class ModelFeature:
    encoder_output: torch.Tensor
    hidden_output: torch.Tensor
    gate: torch.Tensor
