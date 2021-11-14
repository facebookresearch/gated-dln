from __future__ import annotations

import torch

from src.experiment.ds import Task, TasksForKPathModel
from src.task.utils import get_in_and_out_features, get_num_transformations


def get_tasks(
    name: str,
    mode: str,
    num_classes_in_selected_dataset: int,
    num_classes_in_full_dataset: int,
    device: torch.device,
):
    num_input_transformations, num_output_transformations = get_num_transformations(
        mode=mode,
        num_classes_in_selected_dataset=num_classes_in_selected_dataset,
    )
    input_transforms = [lambda x: x for _ in range(num_input_transformations)]
    target_transforms = [lambda x: x for _ in range(num_output_transformations)]

    tasks = []

    for input_index in range(num_input_transformations):
        for output_index in range(num_output_transformations):
            task = Task(
                name=f"{mode}-{input_index}-{output_index}",
                transform=input_transforms[input_index],
                target_transform=target_transforms[output_index],
            )
            tasks.append(task)

    in_features, out_features = get_in_and_out_features(
        dataset_name=name,
        mode=mode,
        num_classes_in_selected_dataset=num_classes_in_selected_dataset,
    )
    return TasksForKPathModel(
        tasks=tasks,
        in_features=in_features,
        out_features=out_features,
        shape=(num_input_transformations, num_output_transformations),
        input_transforms=input_transforms,
        target_transforms=target_transforms,
    )
