from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


class TrainState:
    def __init__(self, num_batches_per_epoch: int, step: int = 0) -> None:

        self.num_batches_per_epoch = num_batches_per_epoch
        self.step = step

    @property
    def epoch(self):
        return self.step // self.num_batches_per_epoch

    @property
    def batch(self):
        return self.step % self.num_batches_per_epoch


class ExperimentMode(Enum):
    TRAIN = "train"
    TEST = "test"


# class MoeMaskMode(Enum):
#     USE_Y_AS_ONE_HOT = "use_y_as_one_hot"
#     AVERAGE = "average"
#     UNIQUE_PATHS_WITH_CONSTRASTIVE_LOSS = "unique_paths_with_constrastive_loss"
#     UNIQUE_PATHS_WITH_COMPETITION = "unique_paths_with_competition"


@dataclass
class ExperimentMetadata:
    mode: ExperimentMode


@dataclass
class Task:
    name: str
    transform: Callable[[torch.Tensor], torch.Tensor]
    target_transform: Callable[[torch.Tensor], torch.Tensor]
