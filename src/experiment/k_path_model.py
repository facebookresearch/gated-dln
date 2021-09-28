"""Class to interface with an Experiment"""
from __future__ import annotations

from time import time
from typing import OrderedDict

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

            tasks = hydra.utils.instantiate(self.cfg.experiment.task)

            self.num_classes_in_original_dataset = (
                self.cfg.experiment.task.num_classes_in_original_dataset
            )

            self.model = instantiate_using_config(
                self.cfg.model,
                tasks=tasks,
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
        inputs = inputs[targets < self.num_classes_in_original_dataset]
        targets = targets[targets < self.num_classes_in_original_dataset]
        loss, loss_to_backprop, num_correct = self.model(
            x=inputs, y=targets, metadata=metadata
        )
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss_to_backprop.backward()  # type: ignore[union-attr]
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
        # key = "matrix"
        # current_metric[f"gating_{key}"] = (self.model.mask, 1)
        # current_metric[f"loss_{key}"] = (loss, total)
        # current_metric[f"accuracy_{key}"] = (num_correct / total, total)

        gate = self.model.gate
        gate_sum = gate.sum()
        flipped_gate = (gate == 0).float()
        flipped_gate_sum = flipped_gate.sum()

        average_accuracy_for_selected_paths = (num_correct * gate).sum() / (
            gate_sum * total
        )
        current_metric["average_accuracy_for_selected_paths"] = (
            average_accuracy_for_selected_paths,
            total,
        )

        average_accuracy_for_unselected_paths = (num_correct * flipped_gate).sum() / (
            flipped_gate_sum * total
        )
        current_metric["average_accuracy_for_unselected_paths"] = (
            average_accuracy_for_unselected_paths,
            total,
        )

        average_loss_for_selected_paths = (loss * gate).sum() / gate_sum
        current_metric["average_loss_for_selected_paths"] = (
            average_loss_for_selected_paths,
            total,
        )

        average_loss_for_unselected_paths = (
            loss * flipped_gate
        ).sum() / flipped_gate_sum
        current_metric["average_loss_for_unselected_paths"] = (
            average_loss_for_unselected_paths,
            total,
        )

        # current_metric["avaerage_loss_for_selected_paths"] =
        if self.should_write_batch_logs:
            metric_to_write = {}
            for key, value in current_metric.items():
                if isinstance(value, (int, float, str)):
                    metric_to_write[key] = value
                else:
                    if isinstance(value[0], (torch.Tensor)):
                        metric_to_write[key] = value[0].detach().cpu().numpy()
                    else:
                        metric_to_write[key] = value[0]
            self.logbook.write_metric(metric=metric_to_write)
        current_metric.pop("time_taken")
        return current_metric

    def train_using_one_dataloader(self) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        buffer = OrderedDict(
            {
                "input": torch.empty(0, 1, 28, 28),
                "target": torch.empty(0, dtype=torch.int64),
            }
        )
        for batch in self.dataloaders[mode]:  # noqa: B007
            input, target = batch
            batch_size = input.shape[0]
            input = input[target < self.num_classes_in_original_dataset]
            target = target[target < self.num_classes_in_original_dataset]
            buffer["input"] = torch.cat([buffer["input"], input], dim=0)
            buffer["target"] = torch.cat([buffer["target"], target], dim=0)
            if buffer["input"].shape[0] >= batch_size:
                batch = [buffer[key][:batch_size] for key in buffer]
                for key in buffer:
                    buffer[key] = buffer[key][batch_size:]
                current_metric = self.compute_metrics_for_batch(
                    batch=batch,
                    mode=mode,
                    batch_idx=self.train_state.batch,
                )
                metric_dict.update(metrics_dict=current_metric)
                self.train_state.step += 1
        if buffer["input"].shape[0] >= batch_size:
            batch = [buffer[key][:batch_size] for key in buffer]
            for key in buffer:
                buffer[key] = buffer[key][batch_size:]
            current_metric = self.compute_metrics_for_batch(
                batch=batch,
                mode=mode,
                batch_idx=self.train_state.batch,
            )
            metric_dict.update(metrics_dict=current_metric)
            self.train_state.step += 1
        metric_dict = metric_dict.to_dict()
        for key in metric_dict:
            if isinstance(metric_dict[key], (torch.Tensor)):
                metric_dict[key] = metric_dict[key].cpu().numpy()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def test_using_one_dataloader(self) -> None:
        epoch_start_time = time()
        self.model.eval()
        mode = "test"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        testloader = self.dataloaders[mode]
        with torch.inference_mode():
            for batch_idx, batch in enumerate(testloader):  # noqa: B007
                input, target = batch
                input = input[target < self.num_classes_in_original_dataset]
                target = target[target < self.num_classes_in_original_dataset]
                if len(input) > 0:
                    current_metric = self.compute_metrics_for_batch(
                        batch=batch, mode=mode, batch_idx=batch_idx
                    )
                    metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        for key in metric_dict:
            if isinstance(metric_dict[key], (torch.Tensor)):
                metric_dict[key] = metric_dict[key].cpu().numpy()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def init_metric_dict(self, epoch: int, mode: str) -> ml_metrics.MetricDict:
        metric_dict = ml_metrics.MetricDict(
            [
                # ml_metrics.AverageMetric("loss_matrix"),
                # ml_metrics.AverageMetric("accuracy_matrix"),
                ml_metrics.AverageMetric("average_loss_for_selected_paths"),
                ml_metrics.AverageMetric("average_loss_for_unselected_paths"),
                ml_metrics.AverageMetric("average_accuracy_for_selected_paths"),
                ml_metrics.AverageMetric("average_accuracy_for_unselected_paths"),
                ml_metrics.CurrentMetric("gating_matrix"),
                ml_metrics.ConstantMetric("epoch", epoch),
                ml_metrics.ConstantMetric("mode", mode),
                ml_metrics.CurrentMetric("batch_index"),
                ml_metrics.CurrentMetric("step"),
            ]
        )

        return metric_dict