import functools
import operator
from collections import UserList
from typing import Any

import torch
from attrs import asdict, define
from torchvision.transforms import functional as functional_transforms


@define
class Transform:
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        return asdict(self)

    def load_state(self, state: dict[str, Any]) -> "Transform":
        return self.__class__(**state)


@define
class IdentityTransform(Transform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


@define
class RotationTransform(Transform):
    angle: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return functional_transforms.rotate(img=x, angle=self.angle)


@define
class PermutationTransform(Transform):
    permuted_indices: torch.Tensor
    product_of_dims: int

    @classmethod
    def build(cls, permuted_indices: torch.Tensor):
        return cls(
            permuted_indices=permuted_indices,
            product_of_dims=functools.reduce(operator.mul, permuted_indices.shape, 1),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, self.product_of_dims)[:, self.permuted_indices].view(
            batch_size, *self.permuted_indices.shape
        )


@define
class MapTransform(Transform):
    input_to_output_map: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_to_output_map[x]


# collection of transforms
class TransformList(UserList):
    def __init__(self, transforms: list[Transform]):
        super().__init__(transforms)

    def get_state(self) -> list[dict[str, Any]]:
        return [transform.get_state() for transform in self.data]

    def load_state(self, state_list: list[dict[str, Any]]) -> "TransformList":
        return TransformList(
            [
                transform.load_state(state)
                for transform, state in zip(self.data, state_list)
            ]
        )


# apply the transforms in the transform list, one after the other
class TransformPath(TransformList):
    def __call__(self, x: torch.Tensor) -> Any:
        return functools.reduce(
            lambda _tensor, transform: transform(_tensor), self.data, x
        )

    @property
    def path_len(self) -> int:
        return len(self.data)


# apply the transforms in the transform list in parallel
class TransformBlock(TransformList):
    def __call__(self, x: torch.Tensor) -> Any:
        return [transform(x) for transform in self.data]

    @property
    def block_size(self) -> int:
        return len(self.data)


# apply the transforms blocks in the transform block list, oner after the other
class PathOfTransformBlocks(UserList):
    def __init__(self, transform_blocks: list[TransformBlock]):
        super().__init__(transform_blocks)

    def __call__(self, x: torch.Tensor) -> Any:
        transformed_x = functools.reduce(
            lambda _tensor, transform_block: torch.cat(
                [
                    tranformed_tensor.unsqueeze(0)
                    for tranformed_tensor in transform_block(_tensor)
                ],
                dim=0,
            ).view(len(transform_block) * _tensor.shape[0], -1),
            self.data,
            x,
        )
        if len(transformed_x.shape) == 2 and transformed_x.shape[-1] == 1:
            # this handles the case where an extra dimension is added to the target tensor
            return transformed_x.squeeze(1)
        return transformed_x

    @property
    def path_len(self) -> int:
        return len(self.data)

    @property
    def block_sizes(self) -> list[int]:
        return [_data.block_size() for _data in self.data]

    def get_state(self) -> list[dict[str, Any]]:
        return [transform_block.get_state() for transform_block in self.data]

    def load_state(self, state_list: list[dict[str, Any]]) -> "PathOfTransformBlocks":
        return PathOfTransformBlocks(
            [
                transform_block.load_state(state_list=state)
                for transform_block, state in zip(self.data, state_list)
            ]
        )
