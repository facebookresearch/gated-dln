# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import torch

from src.data.utils import get_num_input_transformations_from_dataloader_name
from src.ds.task import Task, TasksForKPathModel
from src.task.utils import get_in_and_out_features


def get_tasks(
    name: str,
    mode: str,
    num_classes_in_selected_dataset: int,
    num_classes_in_full_dataset: int,
    num_input_transformations: int,
    num_output_transformations: int,
    device: torch.device,
):
    assert num_input_transformations == num_output_transformations
    assert (
        num_input_transformations
        == get_num_input_transformations_from_dataloader_name(name=name)
    )

    input_transforms = [lambda x: x for _ in range(num_input_transformations)]
    target_transforms = [lambda x: x for _ in range(num_output_transformations)]

    tasks = []

    for input_index in range(num_input_transformations):
        for output_index in range(num_output_transformations):
            task = Task(
                name=f"{mode}-{input_index}-{output_index}",
                index_for_input_transform=input_index,
                index_for_output_transform=output_index,
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
