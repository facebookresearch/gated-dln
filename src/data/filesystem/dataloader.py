from __future__ import annotations

from pathlib import Path

import torch

from src.utils.config import DictConfig, to_dict


def build_dataloaders(
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
    transform,
    target_transform,
    is_preprocessed: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    if not (name == "mnist" or name.startswith("preprocessed_cifar10_dataset_")):
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


def _load_dataset(path: str, mode: str) -> torch.utils.data.TensorDataset:
    features = _load_features_or_labels(
        path=path, mode=mode, features_or_labels="features"
    )
    labels = _load_features_or_labels(path=path, mode=mode, features_or_labels="labels")
    return torch.utils.data.TensorDataset(features, labels)


def _load_features_or_labels(
    path: str, mode: str, features_or_labels: str
) -> torch.Tensor:
    features_or_labels_path = Path(f"{path}/{mode}_{features_or_labels}.pt")
    if features_or_labels_path.exists():
        features_or_labels_tensor = torch.load(features_or_labels_path).to("cpu")
    else:
        features_or_labels_path_list = (
            features_or_labels_path
            for features_or_labels_path in Path(path).iterdir()
            if features_or_labels_path.name.startswith(f"{mode}_{features_or_labels}")
        )

        features_or_labels_tensor = torch.cat(
            [
                torch.load(features_or_labels_path).to("cpu")
                for features_or_labels_path in features_or_labels_path_list
            ],
            dim=0,
        )

    return features_or_labels_tensor


def _build_datasets(
    name: str,
    train_config: DictConfig,
    test_config: DictConfig,
) -> dict[str, torch.utils.data.Dataset]:

    return {
        "train": _load_dataset(
            path=f"{train_config['dataset']['dir']}/{name}", mode="train"
        ),
        "test": _load_dataset(
            path=f"{test_config['dataset']['dir']}/{name}", mode="test"
        ),
    }
