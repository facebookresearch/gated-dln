from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as functional_transforms

from src.task.utils import get_input_shape


def get_rotation_transform(angle: float):
    def transform(x):
        return functional_transforms.rotate(img=x, angle=angle)

    return transform


def get_list_of_rotation_transformations(
    num_transformations: int, full_angle: float = 180.0
) -> list[Any]:
    transforms = []
    for input_index in range(num_transformations):
        angle = full_angle * input_index / (num_transformations)
        transforms.append(get_rotation_transform(angle=angle))
    return transforms


def get_permutation_transform(
    dataset_name: str,
    mode: str,
    num_classes_in_selected_dataset: int,
    seed: int,
    device: torch.device,
):
    rng_permute = np.random.default_rng(seed=seed)
    input_shape = get_input_shape(dataset_name=dataset_name)
    if dataset_name == "mnist":
        dim1, dim2 = input_shape
        assert dim1 == dim2
        permuted_indices = (
            torch.from_numpy(rng_permute.permutation(dim1 * dim2))
            .to(device)
            .view(1, dim1, dim2)
        )

        def transform(x):
            batch_size = x.shape[0]
            return x.view(batch_size, dim1 * dim2)[:, permuted_indices].view(
                batch_size, 1, dim1, dim2
            )

    elif dataset_name == "cifar10":
        num_channels, dim1, dim2 = input_shape
        assert dim1 == dim2

        permuted_indices = (
            torch.from_numpy(rng_permute.permutation(dim1 * dim2))
            .to(device)
            .view(dim1, dim2)
            .unsqueeze(0)
            .repeat(input_shape[0], 1, 1)
        )

        def transform(x):
            batch_size = x.shape[0]
            return x.view(batch_size, num_channels * dim1 * dim2)[
                :, permuted_indices
            ].view(batch_size, num_channels, dim1, dim2)
            # v1 = x.view(batch_size, num_channels, dim1, dim2)
            # v1[:,0] = x.view(batch_size, num_channels, dim1 * dim2)[:,0, permuted_indices[0]]
            # v1[:,1] = x.view(batch_size, num_channels, dim1 * dim2)[:,1, permuted_indices[1]]
            # v1[:,2] = x.view(batch_size, num_channels, dim1 * dim2)[:,2, permuted_indices[2]]

            # v1 = x.view(batch_size, num_channels, dim1 * dim2)[:, :, permuted_indices[1]]
            # v1 = v1[
            #     :, :, permuted_indices[0]
            # ]
            # v2 = x.view(batch_size, num_channels * dim1 * dim2)[:, permuted_indices]

    return transform


def get_list_of_permutation_transformations(
    dataset_name: str,
    mode: str,
    num_classes_in_selected_dataset: int,
    num_transformations: int,
    device: torch.device,
):
    transforms = []
    for input_index in range(num_transformations):
        transforms.append(
            get_permutation_transform(
                dataset_name=dataset_name,
                mode=mode,
                num_classes_in_selected_dataset=num_classes_in_selected_dataset,
                seed=input_index,
                device=device,
            )
        )
    return transforms
