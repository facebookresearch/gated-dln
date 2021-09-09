"""Class to interface with an Experiment"""
from __future__ import annotations

from time import time

import hydra
import torch
import torch.utils.data
from omegaconf.dictconfig import DictConfig
from torchvision import transforms
from xplogger import metrics as ml_metrics
from xplogger.logbook import LogBook
from xplogger.types import LogType

from src.experiment import base as base_experiment
from src.experiment.ds import ExperimentMetadata, ExperimentMode, Task
from src.model.base import Model as BaseModel
from src.utils.config import instantiate_using_config
from src.utils.types import OptimizerType


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


class Experiment(base_experiment.Experiment):
    """Experiment Class"""

    def __init__(
        self,
        cfg: DictConfig,
        logbook: LogBook,
        experiment_id: str = "0",
        should_init: bool = True,
    ):
        """Experiment Class

        Args:
            config (DictConfig):
            logbook (LogBook):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(
            cfg=cfg,
            logbook=logbook,
            experiment_id=experiment_id,
            should_init=False,
        )

        if should_init:

            transform = transforms.ToTensor()

            self.dataloaders: dict[
                str, torch.utils.data.DataLoader
            ] = hydra.utils.instantiate(
                self.cfg.dataloader,
                transform=transform,
                target_transform=None,
            )

            task_one_cfg = self.cfg.experiment.task_one

            task_one = Task(
                name=task_one_cfg.name,
                transform=get_transform(task_one_cfg.transform),
                target_transform=get_target_transform(task_one_cfg.name),
            )

            task_two_cfg = self.cfg.experiment.task_two

            task_two = Task(
                name=task_two_cfg.name,
                transform=get_transform(task_two_cfg.transform),
                target_transform=get_target_transform(task_two_cfg.name),
            )
            in_features = 28 ** 2
            out_features = 2
            self.model = instantiate_using_config(
                self.cfg.model,
                task_one=task_one,
                task_two=task_two,
                in_features=in_features,
                out_features=out_features,
            ).to(self.device)

            assert isinstance(self.model, BaseModel)
            self.optimizer: OptimizerType = hydra.utils.instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )

            self._supported_modes = ["train", "test"]

            self.metadata = {
                mode: ExperimentMetadata(mode=ExperimentMode(mode))
                for mode in self._supported_modes
            }
            self._post_init()

    def compute_metrics_for_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        start_time = time()
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = self.metadata[mode]
        outputs, loss, num_correct = self.model(x=inputs, y=targets, metadata=metadata)
        aggregated_loss = sum(loss.values()) / 2
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            aggregated_loss.backward()  # type: ignore[union-attr]
            # error: Item "float" of "Union[Any, float]" has no attribute "backward"
            self.optimizer.step()
        total = targets.size(0)

        current_metric = {
            "batch_index": batch_idx,
            "epoch": self.train_state.epoch,
            "step": self.train_state.step,
            "time_taken": time() - start_time,
            "mode": mode,
        }
        for key in num_correct:
            current_metric[f"loss_{key}"] = (loss[key].item(), total)
            current_metric[f"accuracy_{key}"] = (num_correct[key] / total, total)
        # breakpoint()
        if self.should_write_batch_logs:
            metric_to_write = {}
            for key, value in current_metric.items():
                if isinstance(value, (int, float, str)):
                    metric_to_write[key] = value
                else:
                    metric_to_write[key] = value[0]
            self.logbook.write_metric(metric=metric_to_write)
        current_metric.pop("time_taken")
        return current_metric

    def init_metric_dict(self, epoch: int, mode: str) -> ml_metrics.MetricDict:
        metric_dict = ml_metrics.MetricDict(
            [
                ml_metrics.AverageMetric("loss_task_one"),
                ml_metrics.AverageMetric("loss_task_two"),
                ml_metrics.AverageMetric("loss_train_on_task_two_eval_on_task_one"),
                ml_metrics.AverageMetric("loss_train_on_task_one_eval_on_task_two"),
                ml_metrics.AverageMetric("loss_task_two_encoder_task_one_decoder"),
                ml_metrics.AverageMetric("loss_task_one_encoder_task_two_decoder"),
                ml_metrics.AverageMetric("accuracy_task_one"),
                ml_metrics.AverageMetric("accuracy_task_two"),
                ml_metrics.AverageMetric("accuracy_train_on_task_two_eval_on_task_one"),
                ml_metrics.AverageMetric("accuracy_train_on_task_one_eval_on_task_two"),
                ml_metrics.AverageMetric("accuracy_task_two_encoder_task_one_decoder"),
                ml_metrics.AverageMetric("accuracy_task_one_encoder_task_two_decoder"),
                ml_metrics.ConstantMetric("epoch", epoch),
                ml_metrics.ConstantMetric("mode", mode),
                ml_metrics.CurrentMetric("batch_index"),
                ml_metrics.CurrentMetric("step"),
            ]
        )

        return metric_dict
