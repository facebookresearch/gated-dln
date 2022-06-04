from __future__ import annotations

from enum import Enum
from typing import Callable

import torch
from attrs import define


@define
class TrainState:
    num_batches_per_epoch: int
    step: int

    @property
    def epoch(self):
        return self.step // self.num_batches_per_epoch

    @property
    def batch(self):
        return self.step % self.num_batches_per_epoch

    def __repr__(self) -> str:
        return f"step: {self.step}, batch: {self.batch}, epoch: {self.epoch}, num_batches_per_epoch: {self.num_batches_per_epoch}"


class ExperimentMode(Enum):
    TRAIN = "train"
    TEST = "test"


@define
class ExperimentMetadata:
    mode: ExperimentMode

    @classmethod
    def build(cls, mode: str) -> ExperimentMetadata:
        return ExperimentMetadata(mode=ExperimentMode(mode))


@define
class Task:
    name: str
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
