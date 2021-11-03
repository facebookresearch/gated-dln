# type: ignore
from __future__ import annotations

from typing import OrderedDict

import hydra
import torch
from omegaconf import DictConfig
from torchvision import transforms as transforms

from src.data.cifar10_v1.feature_extractor import (
    ResNetModel as ResNetModelForFeatureExtraction,
)
from src.data.cifar10_v1.utils import normalize
from src.utils.config import to_dict


def build_task_specific_dataloaders(
    name: str,
    task_specific_cfgs,
    target_transform=None,
) -> dict[str, dict[str, torch.utils.data.DataLoader]]:

    dataloaders = OrderedDict(
        {
            mode: build_dataloaders(
                name=name, target_transform=target_transform, **task_specific_cfgs[mode]
            )
            for mode in task_specific_cfgs
        }
    )
    dataloaders_to_return = {
        mode: [x[mode] for x in dataloaders.values()]
        for mode in list(dataloaders.values())[0]
    }
    return dataloaders_to_return


def build_dataloaders(
    name: str,
    train_config,
    test_config,
    target_transform=None,
) -> dict[str, torch.utils.data.DataLoader]:

    # if transform is None:
    # transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    datasets = _build_datasets(
        train_config=train_config,
        test_config=test_config,
        train_transform=train_transform,
        test_transform=test_transform,
        target_transform=target_transform,
    )
    model_for_feature_extraction_during_train = ResNetModelForFeatureExtraction(
        **train_config.model
    )
    model_for_feature_extraction_during_test = ResNetModelForFeatureExtraction(
        **test_config.model
    )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            **to_dict(train_config.dataloader, resolve=True),
            dataset=datasets["train"],
            collate_fn=model_for_feature_extraction_during_train.forward,
        ),
        "test": torch.utils.data.DataLoader(
            **to_dict(train_config.dataloader, resolve=True),
            dataset=datasets["test"],
            collate_fn=model_for_feature_extraction_during_test.forward,
        ),
    }
    return dataloaders


def _build_datasets(
    train_config: DictConfig,
    test_config: DictConfig,
    train_transform,
    test_transform,
    target_transform,
) -> dict[str, torch.utils.data.Dataset]:
    return {
        "train": hydra.utils.instantiate(
            train_config.dataset,
            transform=train_transform,
            target_transform=target_transform,
        ),
        "test": hydra.utils.instantiate(
            test_config.dataset,
            transform=test_transform,
            target_transform=target_transform,
        ),
    }
