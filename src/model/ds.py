# Copyright (c) Meta Platforms, Inc. and affiliates.
from attrs import define

from src.utils.types import OptimizerType, SchedulerType


@define
class OptimizerSchedulerTuple:
    optimizer: OptimizerType
    scheduler: SchedulerType
