from typing import Dict

import torch
import torchvision
from torchvision import transforms as transforms

from src.utils.config import DictConfig, to_dict


def build_dataloaders(
    name: str,
    train_config,
    test_config,
    transform,
    target_transform,
) -> Dict[str, torch.utils.data.DataLoader]:

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    datasets = _build_datasets(
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
    train_config: DictConfig, test_config: DictConfig, transform, target_transform
) -> Dict[str, torch.utils.data.Dataset]:

    return {
        "train": torchvision.datasets.MNIST(
            **to_dict(train_config.dataset, resolve=True),
            transform=transform,
            target_transform=target_transform,
        ),
        "test": torchvision.datasets.MNIST(
            **to_dict(test_config.dataset, resolve=True),
            transform=transform,
            target_transform=target_transform,
        ),
    }
