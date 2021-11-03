from __future__ import annotations

import itertools

import torch


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

    def transform(y):
        return cache[y]

    return transform


def get_list_of_target_transformations_using_class_combination(
    num_transformations: int,
    num_classes_in_original_dataset: int,
    num_classes_in_full_dataset: int,
    device: torch.device,
):
    class_combinations = list(
        itertools.combinations(
            list(range(num_classes_in_original_dataset)),
            num_classes_in_original_dataset // 2,
        )
    )
    target_transforms = []
    for output_index in range(num_transformations):
        target_transforms.append(
            get_target_transform_using_class_combination(
                num_classes_in_full_dataset=num_classes_in_full_dataset,
                class_combination=class_combinations[output_index],
                device=device,
            )
        )

    return target_transforms


def get_target_transform_using_class_permutation(
    class_permutation: tuple[int, ...], device: torch.device
):

    cache = torch.zeros(len(class_permutation), dtype=torch.long)
    for idx, class_idx in enumerate(class_permutation):
        cache[class_idx] = idx
    cache = cache.to(device)

    def transform(y):
        return cache[y]

    return transform


def get_list_of_target_transformations_using_class_permutation(
    num_transformations: int, num_classes_in_original_dataset: int, device: torch.device
):
    class_permutations = list(
        itertools.permutations(
            list(range(num_classes_in_original_dataset)),
            num_classes_in_original_dataset,
        )
    )
    target_transforms = []
    for output_index in range(num_transformations):
        target_transforms.append(
            get_target_transform_using_class_permutation(
                class_permutation=class_permutations[output_index], device=device
            )
        )

    return target_transforms
