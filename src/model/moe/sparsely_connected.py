# type: ignore

import hydra
import torch
from torch import nn

from src.experiment.ds import ExperimentMode, Metadata, MoeMaskMode
from src.model import base as base_model
from src.model.fully_connected import get_input_output_sizes
from src.model.moe.layer import FeedForward, OneToOneExperts
from src.utils.config import DictConfig


class Model(base_model.Model):
    """Feedforward Model."""

    def __init__(
        self,
        name: str,
        model_cfg: DictConfig,
        loss_fn_cfg: DictConfig,
        should_use_non_linearity: bool,
        description: str = "",
    ):
        super().__init__(name=name, model_cfg=model_cfg, description=description)
        self.model = build_model(
            model_cfg=self.model_cfg, should_use_non_linearity=should_use_non_linearity
        )
        _, output_size = get_input_output_sizes(
            dataset_name=self.model_cfg.dataset_name
        )
        self.gating_network = OneToOneExperts(
            num_tasks=output_size, num_experts=output_size
        )
        if self.model_cfg.classifier.should_share:
            self.classifier = nn.Linear(
                in_features=model_cfg.hidden_size, out_features=output_size
            )
        else:
            self.classifier = build_moe_classifier(
                model_cfg=self.model_cfg,
                should_use_non_linearity=should_use_non_linearity,
            )
        if self.model_cfg.classifier.should_freeze:
            for param in self.classifier.parameters():
                param.requires_grad = False
        self._make_one_hot = lambda x: nn.functional.one_hot(x, num_classes=output_size)
        self.neg_class_weight = 1
        if "neg_class_weight" in loss_fn_cfg:
            self.neg_class_weight = loss_fn_cfg.pop("neg_class_weight")
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, metadata: Metadata
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        moe_output = self.model(x)
        if metadata.moe_mask_mode == MoeMaskMode.UNIQUE_PATHS_WITH_CONSTRASTIVE_LOSS:
            classifier_output = self.classifier(moe_output)
            mask_with_correct_order = self.gating_network(y)
            output_with_correct_order = (
                classifier_output * mask_with_correct_order
            ).sum(dim=0)
            loss = self.loss_fn(
                output_with_correct_order,
                torch.ones_like(y.unsqueeze(1), dtype=torch.float32),
            )

            if metadata.experiment_mode == ExperimentMode.TRAIN:
                permuted_y = (y + torch.randint_like(y, low=1, high=9)) % 10
                permuted_mask = self.gating_network(permuted_y)

                permuted_output = (classifier_output * permuted_mask).sum(dim=0)
                loss += self.neg_class_weight * self.loss_fn(
                    permuted_output,
                    torch.zeros_like(y.unsqueeze(1), dtype=torch.float32),
                )
                loss = loss * 0.5
            predicted = torch.sigmoid(output_with_correct_order)
            output = output_with_correct_order
            num_correct = (predicted > 0.5).sum().item()

        elif metadata.moe_mask_mode == MoeMaskMode.UNIQUE_PATHS_WITH_COMPETITION:
            classifier_output = self.classifier(moe_output)
            mask_with_correct_order = self.gating_network(y)

            if metadata.experiment_mode == ExperimentMode.TRAIN:
                raise NotImplementedError(
                    f"experiment_mode = {metadata.experiment_mode} is not supported when metadata.moe_mask_mode = {metadata.moe_mask_mode}"
                )
                # permuted_y = (y + torch.randint_like(y, low=1, high=9)) % 10
                # permuted_mask = self.gating_network(permuted_y)
                # output_with_correct_order = (
                #     classifier_output * mask_with_correct_order
                # ).sum(dim=0)
                # permuted_output = (classifier_output * permuted_mask).sum(dim=0)
                # loss = self.loss_fn(
                #     output_with_correct_order,
                #     torch.ones_like(y.unsqueeze(1), dtype=torch.float32),
                # ) + self.loss_fn(
                #     permuted_output,
                #     torch.zeros_like(y.unsqueeze(1), dtype=torch.float32),
                # )
                # loss = loss * 0.5
                # predicted = F.sigmoid(output_with_correct_order)
                # output = output_with_correct_order
                # num_correct = (predicted > 0.5).sum().item()

            elif metadata.experiment_mode == ExperimentMode.TEST:
                output, predicted = (
                    x.squeeze() for x in torch.max(classifier_output, dim=0)
                )
                predicted_eq_y = predicted.eq(y)
                num_correct = predicted_eq_y.sum().item()
                loss = self.loss_fn(output, predicted_eq_y.float())

        else:
            if metadata.moe_mask_mode == MoeMaskMode.AVERAGE:
                output = moe_output.mean(dim=0)

            elif metadata.moe_mask_mode == MoeMaskMode.USE_Y_AS_ONE_HOT:
                mask = self.gating_network(y)
                output = (moe_output * mask).sum(dim=0)
            else:
                raise ValueError(
                    f"moe_mask_mode = {metadata.moe_mask_mode} is not supported"
                )
            output = self.classifier(output)
            loss = self.loss_fn(output, y)
            _, predicted = output.max(1)
            num_correct = predicted.eq(y).sum().item()
        assert isinstance(num_correct, int)
        return output, loss, num_correct


def build_model(model_cfg: DictConfig, should_use_non_linearity: bool) -> nn.Module:
    """Build the model

    Args:
        name (str): [description]
        model_cfg (DictConfig): [description]

    Returns:
        nn.Module: [description]
    """
    input_size, output_size = get_input_output_sizes(
        dataset_name=model_cfg.dataset_name
    )
    feedforward_model = FeedForward(
        num_experts=output_size,
        in_features=input_size,
        out_features=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers - 1,
        hidden_features=model_cfg.hidden_size,
        should_use_non_linearity=should_use_non_linearity,
    )
    return nn.Sequential(nn.Flatten(start_dim=1), feedforward_model)


def build_moe_classifier(
    model_cfg: DictConfig, should_use_non_linearity: bool
) -> nn.Module:
    """Build the moe-based classifier

    Args:
        name (str): [description]
        model_cfg (DictConfig): [description]

    Returns:
        nn.Module: [description]
    """
    _, output_size = get_input_output_sizes(dataset_name=model_cfg.dataset_name)
    classifier = FeedForward(
        num_experts=output_size,
        in_features=model_cfg.hidden_size,
        out_features=1,
        num_layers=1,
        hidden_features=model_cfg.hidden_size,
        should_use_non_linearity=should_use_non_linearity,
    )
    return classifier
