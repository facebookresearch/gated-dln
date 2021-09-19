from __future__ import annotations

import itertools

import torch
from torchvision.transforms import functional as functional_transforms

from src.experiment.ds import Task, TasksForKPathModel


def get_rotation_transform(angle: float):
    def transform(x):
        return functional_transforms.rotate(img=x, angle=angle)

    return transform


def get_target_transform(class_combination: list[int], device: torch.device):

    cache = torch.zeros(10, dtype=torch.long)
    for class_idx in class_combination:
        cache[class_idx] = 1
    cache = cache.to(device)

    def transform(y):
        return cache[y]

    return transform


def get_tasks(mode: str, num_classes_in_original_dataset: int, device: str):
    assert mode == "rotate"

    device = torch.device(device)

    class_combinations = list(
        itertools.combinations(
            list(range(num_classes_in_original_dataset)),
            num_classes_in_original_dataset // 2,
        )
    )
    print(class_combinations)
    k = len(class_combinations) // 2
    class_combinations = class_combinations[:k]
    print(class_combinations)
    tasks = []
    input_transforms = []
    target_transforms = []
    for input_index in range(k):
        angle = 90.0 * input_index / (k)
        input_transforms.append(get_rotation_transform(angle=angle))

    for output_index in range(k):
        target_transforms.append(
            get_target_transform(
                class_combination=class_combinations[output_index], device=device
            )
        )

    for input_index in range(k):
        for output_index in range(k):
            task = Task(
                name=f"{mode}-{input_index}-{output_index}-{angle}",
                transform=input_transforms[input_index],
                target_transform=target_transforms[output_index],
            )
            tasks.append(task)

    in_features = 28 ** 2
    out_features = 2
    return TasksForKPathModel(
        tasks=tasks,
        in_features=in_features,
        out_features=out_features,
        shape=(k, k),
        input_transforms=input_transforms,
        target_transforms=target_transforms,
    )
