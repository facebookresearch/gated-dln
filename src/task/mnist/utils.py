from __future__ import annotations

import itertools


def get_in_and_out_features(
    mode: str, num_classes_in_selected_dataset: int
) -> tuple[int, int]:
    in_features = 28 ** 2
    if mode.endswith("permute_target"):
        out_features = num_classes_in_selected_dataset
    else:
        out_features = 2
    return (in_features, out_features)


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
