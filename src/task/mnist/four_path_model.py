from __future__ import annotations

from omegaconf import DictConfig

from src.experiment.ds import Task, TasksForFourPathModel


def get_transform(mode):
    if mode == "invert":

        def transform(x):
            return 1 - x

    elif mode == "default":

        def transform(x):
            return x

    else:
        raise ValueError(f"mode={mode} is not defined.")

    return transform


def get_target_transform(name: str):
    if name == "odd_even":

        def transform(y):
            return (y % 2 == 0).long()

    elif name == "greater_than_four":

        def transform(y):
            return (y > 4).long()

    else:
        raise ValueError(f"name={name} is not defined.")

    return transform


def get_tasks(task_one_cfg: DictConfig, task_two_cfg: DictConfig):

    task_one = Task(
        name=task_one_cfg.name,
        transform=get_transform(task_one_cfg.transform),
        target_transform=get_target_transform(task_one_cfg.name),
    )

    task_two = Task(
        name=task_two_cfg.name,
        transform=get_transform(task_two_cfg.transform),
        target_transform=get_target_transform(task_two_cfg.name),
    )
    in_features = 28 ** 2
    out_features = 2

    return TasksForFourPathModel(
        task_one=task_one,
        task_two=task_two,
        in_features=in_features,
        out_features=out_features,
    )
