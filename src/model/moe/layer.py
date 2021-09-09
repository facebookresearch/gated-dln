# Taken from https://github.com/fairinternal/mtrl/blob/master/mtrl/agent/components/moe_layer.py
from __future__ import annotations

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, bias: bool = True
    ):
        """torch.nn.Linear layer extended for use as a mixture of experts.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.in_features, self.out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))
            self.use_bias = True
        else:
            self.use_bias = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias:
            return x.matmul(self.weight) + self.bias
        else:
            return x.matmul(self.weight)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class FunctionalLinear(nn.Module):
    def __init__(self):
        """Functional version of Linear"""
        super().__init__()

    def forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        # shape of weight is (num_experts, in_features, out_features)
        # shape of bias is (num_experts, 1, out_features)

        return input.matmul(weight) + bias

    def extra_repr(self) -> str:
        return "FunctionalLinear layer"


class FeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        should_use_non_linearity: bool,
        bias: bool = True,
    ):
        """A feedforward model of mixture of experts layers.

        Args:
            num_experts (int): number of experts in the mixture.
            in_features (int): size of each input sample for one expert.
            out_features (int): size of each output sample for one expert.
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            bias (bool, optional): if set to ``False``, the layer will
                not learn an additive bias. Defaults to True.
        """
        super().__init__()
        layers: list[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(linear)
            if should_use_non_linearity:
                layers.append(nn.ReLU())
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def __repr__(self) -> str:
        return str(self._model)


class MaskCache:
    def __init__(
        self,
        task_index_to_mask: torch.Tensor,
    ):
        """In multitask learning, using a mixture of models, different tasks
            can be mapped to different combination of models. This utility
            class caches these mappings so that they do not have to be revaluated.

            For example, when the model is training over 10 tasks, and the
            tasks are always ordered, the mapping of task index to encoder indices
            will be the same and need not be recomputed. We take a very simple
            approach here: cache using the number of tasks, since in our case,
            the task ordering during training and evaluation does not change.
            In more complex cases, a mode (train/eval..) based key could be used.

            This gets a little trickier during evaluation. We assume that we are
            running multiple evaluation episodes (per task) at once. So during
            evaluation, the agent is inferring over num_tasks*num_eval_episodes
            at once.

            We have to be careful about not caching the mapping during update because
            neither the task distribution, nor the task ordering, is pre-determined
            during update. So we explicitly exclude the `batch_size` from the list
            of keys being cached.

        Args:
            num_tasks (int): number of tasks.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            task_index_to_mask (torch.Tensor): mapping of task index to mask.
        """
        self.masks: dict[int, torch.Tensor] = {}
        self.task_index_to_mask = task_index_to_mask

    def get_mask(self, task_index: torch.Tensor) -> torch.Tensor:
        return self._make_mask(task_index=task_index)

    def _make_mask(self, task_index: torch.Tensor):
        encoder_mask = self.task_index_to_mask[task_index.squeeze()]
        if len(encoder_mask.shape) == 1:
            encoder_mask = encoder_mask.unsqueeze(0)
        return encoder_mask.to(task_index.device, non_blocking=True).t().unsqueeze(2)


class MixtureOfExperts(nn.Module):
    def __init__(self):
        """Class for interfacing with a mixture of experts."""
        super().__init__()
        self.mask_cache: MaskCache

    def forward(self, task_index: torch.Tensor) -> torch.Tensor:
        return self.mask_cache.get_mask(task_index=task_index)


class OneToOneExperts(MixtureOfExperts):
    def __init__(
        self,
        num_tasks: int,
        num_experts: int,
    ):
        """Map the output of ith expert with the ith task.

        Args:
            num_tasks (int): number of tasks.
            num_experts (int): number of experts in the mixture of experts.
            num_eval_episodes (int): number of episodes run during evaluation.
            batch_size (int): batch size for update.
            multitask_cfg (DictConfig): config for multitask training.
        """
        super().__init__()
        assert num_tasks == num_experts
        self.mask_cache = MaskCache(
            task_index_to_mask=torch.eye(num_tasks),
        )
