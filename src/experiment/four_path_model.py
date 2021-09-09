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
from src.experiment.ds import ExperimentMetadata, ExperimentMode
from src.model.base import Model as BaseModel
from src.utils.config import instantiate_using_config
from src.utils.types import OptimizerType


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
            if self.should_use_task_specific_dataloaders:
                self.dataloaders: dict[
                    str, dict[str, torch.utils.data.DataLoader]
                ] = hydra.utils.instantiate(self.cfg.dataloader, target_transform=None)

            else:
                self.dataloaders: dict[
                    str, torch.utils.data.DataLoader
                ] = hydra.utils.instantiate(
                    self.cfg.dataloader,
                    transform=transform,
                    target_transform=None,
                )

            tasks = hydra.utils.instantiate(
                self.cfg.experiment.task,
                task_one_cfg=self.cfg.experiment.task_one,
                task_two_cfg=self.cfg.experiment.task_two,
            )

            self.model = instantiate_using_config(
                self.cfg.model,
                task_one=tasks.task_one,
                task_two=tasks.task_two,
                in_features=tasks.in_features,
                out_features=tasks.out_features,
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
        if self.should_use_task_specific_dataloaders:
            total = targets[0].size(0)
        else:
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
