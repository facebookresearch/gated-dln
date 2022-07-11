# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import warnings

import torch
from torchvision import transforms as transforms

from src.data.torchvision import dataloader as torchvision_data_utils
from src.utils.config import DictConfig


def build_dataloaders(
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
    transform,
    target_transform,
    is_preprocessed: bool,
) -> dict[str, torch.utils.data.DataLoader]:

    if transform:
        warnings.warn("transform arg is deprecated", DeprecationWarning, stacklevel=2)

    # todo: remove transform from the function call here.
    transform = transforms.Compose([transforms.ToTensor()])

    return torchvision_data_utils.build_dataloaders(
        name=name,
        train_config=train_config,
        test_config=test_config,
        transform=transform,
        target_transform=target_transform,
        is_preprocessed=is_preprocessed,
    )
