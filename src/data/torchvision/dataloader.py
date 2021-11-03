from __future__ import annotations

from typing import Any

import torch
import torchvision
from torchvision import transforms as transforms

from src.utils.config import DictConfig, to_dict


def build_dataloaders(
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
    transform,
    target_transform,
) -> dict[str, torch.utils.data.DataLoader]:

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    if name == "mnist":
        torchvision_cls = torchvision.datasets.MNIST
    elif name == "cifar10":
        torchvision_cls = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f"name={name} is not supported.")

    datasets = _build_datasets(
        torchvision_cls=torchvision_cls,
        train_config=train_config,
        test_config=test_config,
        transform=transform,
        target_transform=target_transform,
    )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            **to_dict(train_config.dataloader, resolve=True),
            dataset=datasets["train"],
        ),
        "test": torch.utils.data.DataLoader(
            **to_dict(train_config.dataloader, resolve=True),
            dataset=datasets["test"],
        ),
    }
    return dataloaders


def _build_datasets(
    torchvision_cls: Any,
    train_config: DictConfig,
    test_config: DictConfig,
    transform,
    target_transform,
) -> dict[str, torch.utils.data.Dataset]:

    return {
        "train": torchvision_cls(
            **to_dict(train_config.dataset, resolve=True),
            transform=transform,
            target_transform=target_transform,
        ),
        "test": torchvision_cls(
            **to_dict(test_config.dataset, resolve=True),
            transform=transform,
            target_transform=target_transform,
        ),
    }
