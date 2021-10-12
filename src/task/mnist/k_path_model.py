from __future__ import annotations

from src.experiment.ds import Task, TasksForKPathModel
from src.task.mnist.input_transforms import (
    get_list_of_permutation_transformations,
    get_list_of_rotation_transformations,
)
from src.task.mnist.target_transforms import (
    get_list_of_target_transformations_using_class_combination,
    get_list_of_target_transformations_using_class_permutation,
)
from src.task.mnist.utils import get_in_and_out_features, get_num_transformations


def get_tasks(mode: str, num_classes_in_original_dataset: int, device: str):
    num_input_transformations, num_output_transformations = get_num_transformations(
        mode=mode, num_classes_in_original_dataset=num_classes_in_original_dataset
    )
    if mode in ["rotate"] or mode.startswith("rotate_input"):
        input_transforms = get_list_of_rotation_transformations(
            num_transformations=num_input_transformations
        )
    elif mode in ["permute"] or mode.startswith("permute_input"):
        input_transforms = get_list_of_permutation_transformations(
            mode=mode,
            num_classes_in_original_dataset=num_classes_in_original_dataset,
            num_transformations=num_input_transformations,
            device=device,
        )
    else:
        raise NotImplementedError(f"mode={mode} is not supported.")

    if mode in ["rotate"] or mode.endswith("rotate_input"):
        breakpoint()
        target_transforms = get_list_of_target_transformations_using_class_combination(
            num_transformations=num_input_transformations,
            num_classes_in_original_dataset=num_classes_in_original_dataset,
            device=device,
        )
    elif mode.endswith("permute_target"):
        target_transforms = get_list_of_target_transformations_using_class_permutation(
            num_transformations=num_input_transformations,
            num_classes_in_original_dataset=num_classes_in_original_dataset,
            device=device,
        )
    else:
        raise NotImplementedError(f"mode={mode} is not supported.")

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
        mode=mode, num_classes_in_original_dataset=num_classes_in_original_dataset
    )
    return TasksForKPathModel(
        tasks=tasks,
        in_features=in_features,
        out_features=out_features,
        shape=(num_input_transformations, num_output_transformations),
        input_transforms=input_transforms,
        target_transforms=target_transforms,
    )
