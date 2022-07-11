# Copyright (c) Meta Platforms, Inc. and affiliates.
# type: ignore
"""Collection of CIFAR datasets that are grouped by classes. For more details refer #28"""


from typing import Callable, Optional, OrderedDict

import torchvision

from src.data.cifar10_v1.utils import labels_map


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        labels_to_retain: list[str],
        labels_to_group: list[list[str]],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        filter_fn=lambda x: True,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if labels_to_retain:
            # labels_to_retain = set(labels_to_retain)
            if not labels_to_group:
                labels_to_group_map = OrderedDict(
                    {label: label for label in labels_to_retain}
                )
            else:
                labels_to_group_map = OrderedDict({})
                for current_labels_to_group in labels_to_group:
                    for current_label in current_labels_to_group:
                        labels_to_group_map[current_label] = current_labels_to_group[0]
            filtered_data, filtered_targets = [], []
            target_mapping = {
                label: idx for idx, label in enumerate(labels_to_group_map.keys())
            }
            idx = 0
            for x, y in zip(self.data, self.targets):
                if labels_map[y] not in labels_to_group_map:
                    continue
                y_label = labels_to_group_map[labels_map[y]]
                if y_label in labels_to_retain and filter_fn(idx):
                    filtered_data.append(x)
                    # if y_label not in target_mapping:
                    # target_mapping[y_label] = len(target_mapping)
                    filtered_targets.append(target_mapping[y_label])
                idx += 1
            self.data = filtered_data
            self.targets = filtered_targets


# def build_cifar10_airplane_automobile_vs_bird_cat(
#     root: str,
#     train: bool = True,
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     download: bool = False,
# ):
#     labels_to_retain: list[str] = ["airplane", "automobile", "bird", "cat"]
#     labels_to_group: list[list[str]] = [
#         ["airplane", "automobile"],
#         ["bird", "cat"],
#     ]
#     return CustomCIFAR10(
#         root=root,
#         labels_to_retain=labels_to_retain,
#         labels_to_group=labels_to_group,
#         train=train,
#         transform=transform,
#         target_transform=target_transform,
#         download=download,
#     )


# def build_cifar10_airplane_bird_vs_automobile_cat(
#     root: str,
#     train: bool = True,
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     download: bool = False,
# ):
#     labels_to_retain: list[str] = ["airplane", "automobile", "bird", "cat"]
#     labels_to_group: list[list[str]] = [
#         ["airplane", "bird"],
#         ["automobile", "cat"],
#     ]
#     return CustomCIFAR10(
#         root=root,
#         labels_to_retain=labels_to_retain,
#         labels_to_group=labels_to_group,
#         train=train,
#         transform=transform,
#         target_transform=target_transform,
#         download=download,
#     )


def build_cifar10_living_or_not(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
):
    labels_to_retain: list[str] = ["airplane", "automobile", "bird", "cat"]
    labels_to_group: list[list[str]] = []
    #     ["airplane", "automobile"],
    #     ["bird", "cat"],
    # ]
    return CustomCIFAR10(
        root=root,
        labels_to_retain=labels_to_retain,
        labels_to_group=labels_to_group,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        filter_fn=lambda x: x % 2 == 0,
    )


def build_cifar10_fly_or_not(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
):
    labels_to_retain: list[str] = ["airplane", "automobile", "bird", "cat"]
    labels_to_group: list[list[str]] = []
    #     ["automobile", "cat"],
    #     ["airplane", "bird"],
    # ]
    return CustomCIFAR10(
        root=root,
        labels_to_retain=labels_to_retain,
        labels_to_group=labels_to_group,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        filter_fn=lambda x: x % 2 == 1,
    )
