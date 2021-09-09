"""Collection of types used in the code."""

# from typing import Callable
from typing import TYPE_CHECKING

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

OptimizerType = torch.optim.Optimizer
# if TYPE_CHECKING:
#     OptimizerType = torch.optim.optimizer.Optimizer
# else:
#     OptimizerType = torch.optim.Optimizer


SchedulerType = optim.lr_scheduler._LRScheduler

# LossFunctionType = Callable[[TensorType, TensorType], TensorType]


if TYPE_CHECKING:
    DataLoaderType = DataLoader[torch.Tensor]
    DatasetType = Dataset[torch.Tensor]
else:
    DataLoaderType = DataLoader
    DatasetType = Dataset
