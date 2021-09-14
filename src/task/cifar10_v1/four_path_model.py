from __future__ import annotations

from omegaconf import DictConfig

from src.experiment.ds import Task, TasksForFourPathModel


def get_transform(mode):
    return lambda x: x


def get_target_transform(name: str):
    if name == "living_or_not":

        def transform(y):
            return (y // 2 == 0).long()

    elif name == "fly_or_not":

        def transform(y):
            return ((y + 1) % 2).long()

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
    in_features = 2048
    out_features = 2
    return TasksForFourPathModel(
        task_one=task_one,
        task_two=task_two,
        in_features=in_features,
        out_features=out_features,
    )
