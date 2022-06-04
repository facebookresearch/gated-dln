from __future__ import annotations

import torch

from src.task.k_path.cifar10 import builder as cifar10_builder
from src.task.k_path.cifar10_preprocessed import builder as cifar10_preprocessed_builder
from src.task.k_path.mnist import builder as mnist_builder


def get_tasks(
    name: str,
    mode: str,
    num_classes_in_selected_dataset: int,
    num_classes_in_full_dataset: int,
    num_input_transformations: int,
    num_output_transformations: int,
    device: torch.device,
):
    kwargs = dict(  # noqa: C408
        name=name,
        mode=mode,
        num_classes_in_selected_dataset=num_classes_in_selected_dataset,
        num_classes_in_full_dataset=num_classes_in_full_dataset,
        num_input_transformations=num_input_transformations,
        num_output_transformations=num_output_transformations,
        device=device,
    )
    if name == "mnist":
        return mnist_builder.get_tasks(**kwargs)
    elif name == "cifar10":
        return cifar10_builder.get_tasks(**kwargs)
    elif name.startswith("cifar100"):
        breakpoint()
    elif name.startswith("preprocessed_cifar10_dataset_"):
        return cifar10_preprocessed_builder.get_tasks(**kwargs)
    else:
        breakpoint()
