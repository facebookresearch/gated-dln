from __future__ import annotations

from enum import Enum

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
