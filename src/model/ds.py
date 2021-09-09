from dataclasses import dataclass

from src.utils.types import OptimizerType, SchedulerType


@dataclass
class OptimizerSchedulerTuple:
    optimizer: OptimizerType
    scheduler: SchedulerType
