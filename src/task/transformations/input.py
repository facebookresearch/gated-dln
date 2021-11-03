from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as functional_transforms

from src.task.mnist.utils import get_in_and_out_features


def get_rotation_transform(angle: float):
    def transform(x):
        return functional_transforms.rotate(img=x, angle=angle)

    return transform


def get_list_of_rotation_transformations(
    num_transformations: int, full_angle: float = 90.0
) -> list[Any]:
    transforms = []
    for input_index in range(num_transformations):
        angle = full_angle * input_index / (num_transformations)
        transforms.append(get_rotation_transform(angle=angle))
    return transforms


def get_permutation_transform(
    mode: str, num_classes_in_selected_dataset: int, seed: int, device: torch.device
):
    rng_permute = np.random.default_rng(seed=seed)
    in_features, _ = get_in_and_out_features(
        mode=mode, num_classes_in_selected_dataset=num_classes_in_selected_dataset
    )
    dim = int(math.sqrt(in_features))
    permuted_indices = (
        torch.from_numpy(rng_permute.permutation(in_features))
        .to(device)
        .view(1, dim, dim)
    )

    def transform(x):
        batch_size = x.shape[0]
        return x.view(batch_size, in_features)[:, permuted_indices].view(
            batch_size, 1, dim, dim
        )

    return transform


def get_list_of_permutation_transformations(
    mode: str,
    num_classes_in_selected_dataset: int,
    num_transformations: int,
    device: torch.device,
):
    transforms = []
    for input_index in range(num_transformations):
        transforms.append(
            get_permutation_transform(
                mode=mode,
                num_classes_in_selected_dataset=num_classes_in_selected_dataset,
                seed=input_index,
                device=device,
            )
        )
    return transforms
