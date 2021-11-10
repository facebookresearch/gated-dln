from __future__ import annotations

import torch
from torchvision import transforms as transforms

from src.utils.config import DictConfig, to_dict


def build_dataloaders(
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
    transform,
    target_transform,
) -> dict[str, torch.utils.data.DataLoader]:

    if name not in ["cifar_dataset_6_classes_input_permuted_output_permuted_v1"]:
        raise ValueError(f"name={name} is not supported.")

    datasets = _build_datasets(
        name=name,
        train_config=train_config,
        test_config=test_config,
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
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
) -> dict[str, torch.utils.data.Dataset]:
    return {
        "train": torch.utils.data.TensorDataset(
            torch.load(f"{train_config['dataset']['dir']}/{name}/train_features.pt").to("cpu"),
            torch.load(f"{train_config['dataset']['dir']}/{name}/train_labels.pt").to("cpu")
        ),
        "test": torch.utils.data.TensorDataset(
            torch.load(f"{test_config['dataset']['dir']}/{name}/test_features.pt").to("cpu"),
            torch.load(f"{test_config['dataset']['dir']}/{name}/test_labels.pt").to("cpu")
        ),
    }
