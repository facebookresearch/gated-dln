from __future__ import annotations

import itertools
from functools import reduce


def get_in_and_out_features(
    dataset_name: str, mode: str, num_classes_in_selected_dataset: int
) -> tuple[int, int]:
    input_shape = get_input_shape(dataset_name=dataset_name)
    if mode.endswith("permute_target"):
        out_features = num_classes_in_selected_dataset
    else:
        out_features = 2
    return (reduce(lambda x, y: x * y, input_shape, 1), out_features)


def get_input_shape(dataset_name: str) -> tuple[int, ...]:
    input_shape: tuple[int, ...]
    if dataset_name == "cifar10":
        input_shape = (3, 32, 32)
    elif dataset_name == "mnist":
        input_shape = (28, 28)
    else:
        raise ValueError(f"dataset_name={dataset_name} is not supported.")
    return input_shape


def get_num_transformations(
    mode: str, num_classes_in_selected_dataset: int
) -> tuple[int, int]:
    class_combinations = list(
        itertools.combinations(
            list(range(num_classes_in_selected_dataset)),
            num_classes_in_selected_dataset // 2,
        )
    )
    num_transformations = len(class_combinations) // 2

    return (num_transformations, num_transformations)
