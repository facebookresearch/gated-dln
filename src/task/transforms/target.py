import itertools

import torch
from omegaconf import DictConfig

from src.ds.transform import OutputTransformationMode
from src.task.transforms import ds as task_transforms


def get_target_transform_using_class_combination(
    num_classes_in_full_dataset: int,
    class_combination: tuple[int, ...],
    device: torch.device,
):
    # todo: reproduce some previous results.
    cache = torch.zeros(num_classes_in_full_dataset, dtype=torch.long)
    for class_idx in class_combination:
        cache[class_idx] = 1
    cache = cache.to(device)

    return task_transforms.MapTransform(input_to_output_map=cache)


def get_list_of_target_transformations_using_class_combination(
    num_transformations: int,
    num_selected_classes: int,
    num_classes_in_full_dataset: int,
    device: torch.device,
):
    class_combinations = list(
        itertools.combinations(
            list(range(num_selected_classes)),
            num_selected_classes // 2,
        )
    )
    transforms = []
    for output_index in range(num_transformations):
        transforms.append(
            get_target_transform_using_class_combination(
                num_classes_in_full_dataset=num_classes_in_full_dataset,
                class_combination=class_combinations[output_index],
                device=device,
            )
        )

    return task_transforms.TransformList(transforms)


def get_target_transform_using_class_permutation(
    class_permutation: tuple[int, ...], device: torch.device
):

    cache = torch.zeros(len(class_permutation), dtype=torch.long)
    for idx, class_idx in enumerate(class_permutation):
        cache[class_idx] = idx
    cache = cache.to(device)

    return task_transforms.MapTransform(input_to_output_map=cache)


def get_list_of_target_transformations_using_class_permutation(
    num_transformations: int, num_selected_classes: int, device: torch.device
):
    class_permutations = list(
        itertools.permutations(
            list(range(num_selected_classes)),
            num_selected_classes,
        )
    )
    transforms = []
    for output_index in range(num_transformations):
        transforms.append(
            get_target_transform_using_class_permutation(
                class_permutation=class_permutations[output_index], device=device
            )
        )

    return task_transforms.TransformList(transforms)


def build_list_of_paths_of_transform_blocks(
    transformation_cfg: DictConfig,
) -> list[task_transforms.PathOfTransformBlocks]:
    assert transformation_cfg.block_size == 1
    assert transformation_cfg.path_len == 1
    # if (
    #     transformation_cfg.mode
    #     == OutputTransformationMode.MAP_USING_CLASS_COMBINATION.value
    # ):
    #     transforms = get_list_of_target_transformations_using_class_combination(
    #         num_transformations=transformation_cfg.num_transformations,
    #         num_selected_classes=transformation_cfg.num_selected_classes,
    #         device=transformation_cfg.device,
    #         num_classes_in_full_dataset=transformation_cfg.num_classes_in_full_dataset,
    #     )
    if (
        transformation_cfg.mode
        == OutputTransformationMode.MAP_USING_CLASS_PERMUTATION.value
    ):
        transforms = get_list_of_target_transformations_using_class_permutation(
            num_transformations=transformation_cfg.num_transformations,
            num_selected_classes=transformation_cfg.num_selected_classes,
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
