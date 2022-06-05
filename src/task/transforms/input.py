from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from src.ds.transform import InputTransformationMode
from src.task.transforms import ds as task_transforms
from src.task.utils import get_input_shape


def get_rotation_transform(angle: float):
    return task_transforms.RotationTransform(angle=angle)


def get_list_of_rotation_transformations(
    num_transformations: int, full_angle: float = 180.0
) -> list[Any]:
    transforms = []
    for input_index in range(num_transformations):
        angle = full_angle * input_index / (num_transformations)
        transforms.append(get_rotation_transform(angle=angle))
    return task_transforms.TransformList(transforms)


def get_permutation_transform(
    dataset_name: str,
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

    elif dataset_name == "cifar10":
        _, dim1, dim2 = input_shape
        assert dim1 == dim2

        permuted_indices = (
            torch.from_numpy(rng_permute.permutation(dim1 * dim2))
            .to(device)
            .view(dim1, dim2)
            .unsqueeze(0)
            .repeat(input_shape[0], 1, 1)
        )

    return task_transforms.PermutationTransform.build(permuted_indices=permuted_indices)


def get_list_of_permutation_transformations(
    dataset_name: str,
    num_transformations: int,
    device: torch.device,
):
    transforms = []
    for input_index in range(num_transformations):
        transforms.append(
            get_permutation_transform(
                dataset_name=dataset_name,
                seed=input_index,
                device=device,
            )
        )
    return task_transforms.TransformList(transforms)


def build_list_of_paths_of_transform_blocks(
    transformation_cfg: DictConfig,
) -> list[task_transforms.PathOfTransformBlocks]:
    assert transformation_cfg.block_size == 1
    assert transformation_cfg.path_len == 1
    if transformation_cfg.mode == InputTransformationMode.ROTATE.value:
        transforms = get_list_of_rotation_transformations(
            num_transformations=transformation_cfg.num_transformations,
            full_angle=180,
        )
    elif transformation_cfg.mode == InputTransformationMode.PERMUTE.value:
        transforms = get_list_of_permutation_transformations(
            dataset_name=transformation_cfg.dataset_name,
            num_transformations=transformation_cfg.num_transformations,
            device=transformation_cfg.device,
        )
    else:
        raise NotImplementedError(f"mode={transformation_cfg.mode} is not supported.")
    return [
        task_transforms.PathOfTransformBlocks(
            [task_transforms.TransformBlock([_transform])]
        )
        for _transform in transforms
    ]
