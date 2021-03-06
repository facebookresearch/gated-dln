# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Class to interface with an Experiment"""
from __future__ import annotations

from time import time
from typing import OrderedDict, cast

import hydra
import torch
import torch.utils.data
from functorch import combine_state_for_ensemble
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from xplogger import metrics as ml_metrics
from xplogger.logbook import LogBook
from xplogger.types import LogType

from src.ds.experiment import ExperimentMetadata, TrainState
from src.experiment import base as base_experiment
from src.model.base import Model as BaseModel
from src.utils.config import instantiate_using_config
from src.utils.types import OptimizerType
from src.utils.utils import make_dir


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
            cfg (DictConfig):
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
                # dont recall why we have these.
                self.dataloaders = hydra.utils.instantiate(
                    self.cfg.dataloader, target_transform=None
                )

            else:
                self.dataloaders = hydra.utils.instantiate(
                    self.cfg.dataloader,
                    transform=transform,
                    target_transform=None,
                )
                # Attribute "dataloaders" already defined on line 53  [no-redef]

            self.tasks = hydra.utils.instantiate(self.cfg.experiment.task)

            self.num_classes_in_selected_dataset = (
                self.cfg.experiment.task.num_classes_in_selected_dataset
            )
            self.model = instantiate_using_config(
                self.cfg.model,
                tasks=self.tasks,
            ).to(self.device)

            assert isinstance(self.model, BaseModel)
            self.optimizer: OptimizerType = hydra.utils.instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )

            self._supported_modes = ["train", "test"]

            self.metadata = {
                mode: ExperimentMetadata.build(mode=mode)
                for mode in self._supported_modes
            }

            if self.cfg.model.hidden_layer_cfg.should_share:
                self.compute_metrics_for_batch = (  # type: ignore[assignment]
                    self.compute_metrics_for_batch_when_using_shared_hidden_layer
                )
                # Cannot assign to a method  [assignment]
                self.compute_metrics_for_batch_for_eval = (  # type: ignore[assignment]
                    self.compute_metrics_for_batch_for_eval_when_using_shared_hidden_layer
                )
                # Cannot assign to a method  [assignment]
            else:
                self.compute_metrics_for_batch = (  # type: ignore[assignment]
                    self.compute_metrics_for_batch_without_share_hidden
                )
                # Cannot assign to a method  [assignment]
                self.compute_metrics_for_batch_for_eval = (  # type: ignore[assignment]
                    self.compute_metrics_for_batch_without_share_hidden
                )
                # Cannot assign to a method  [assignment]

            self._post_init()

    def _make_train_state(self, start_step: int) -> TrainState:

        scaling_ratio = (
            self.cfg.experiment.task.num_classes_in_selected_dataset
            / self.cfg.experiment.task.num_classes_in_full_dataset
        )

        if self.should_use_task_specific_dataloaders:
            assert isinstance(self.dataloaders["train"], tuple)
            train_state = TrainState(
                num_batches_per_epoch=int(
                    len(self.dataloaders["train"][0]) * scaling_ratio
                ),
                step=start_step,
            )
        else:
            train_state = TrainState(
                num_batches_per_epoch=int(
                    len(self.dataloaders["train"]) * scaling_ratio
                ),
                step=start_step,
            )
        return train_state

    def compute_metrics_for_batch_when_using_shared_hidden_layer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        start_time = time()
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        targets = targets.long()
        loss, loss_to_backprop, num_correct = self.model(x=inputs, y=targets)
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss_to_backprop.backward()
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
        gate_sum = gate.sum()  # type: ignore[operator]
        # "Tensor" not callable  [operator]
        flipped_gate = (gate == 0).float()  # type: ignore[union-attr]
        # Item "bool" of "Union[Tensor, bool]" has no attribute "float"  [union-attr]
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

    def compute_metrics_for_batch_for_eval_when_using_shared_hidden_layer(
        self,
        encoder_fmodel,
        encoder_params,
        encoder_buffers,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        start_time = time()
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        targets = targets.long()
        # metadata = self.metadata[mode]
        (loss, loss_to_backprop, num_correct) = self.model.forward_eval(
            encoder_fmodel=encoder_fmodel,
            encoder_params=encoder_params,
            encoder_buffers=encoder_buffers,
            x=inputs,
            y=targets,
        )
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss_to_backprop.backward()
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
        gate_sum = gate.sum()  # type: ignore[operator]
        # "Tensor" not callable  [operator]
        flipped_gate = (gate == 0).float()  # type: ignore[union-attr]
        # Item "bool" of "Union[Tensor, bool]" has no attribute "float"  [union-attr]
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

    def extract_features_from_one_batch_for_caching_dataset(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = self.metadata[mode]
        features, targets = self.model.extract_features_for_caching_dataset(  # type: ignore[operator]
            x=inputs, y=targets, metadata=metadata
        )
        # error: "Tensor" not callable
        return features, targets

    def _get_input_shape(self) -> tuple[int, ...]:
        if self.cfg.dataloader.name == "mnist":
            return (1, 28, 28)
        elif self.cfg.dataloader.name == "cifar10":
            return (3, 32, 32)
        elif (
            self.cfg.dataloader.name
            == "cifar_dataset_6_classes_input_permuted_output_permuted_v1"
        ):
            return (512,)
        else:
            raise ValueError(
                f"dataloader_name={self.cfg.dataloader.name} is not supported."
            )

    def train_using_one_dataloader_when_using_unprocessed_dataset(self) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        buffer = OrderedDict(
            {
                "input": torch.empty(0, *self._get_input_shape()),
                "target": torch.empty(0, dtype=torch.int64),
            }
        )
        train_dataloader = cast(DataLoader, self.dataloaders[mode])
        for batch in train_dataloader:  # noqa: B007
            input, target = batch
            batch_size = input.shape[0]
            input = input[target < self.num_classes_in_selected_dataset]
            target = target[target < self.num_classes_in_selected_dataset]
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

    def train_using_one_dataloader_when_using_preprocessed_dataset(self) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        train_dataloader = cast(DataLoader, self.dataloaders[mode])
        for batch in train_dataloader:  # noqa: B007
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

    def extract_features_for_caching_dataset(self) -> None:
        path = f"{self.cfg.setup.base_path}/data/processed/{self.cfg.setup.id}"
        make_dir(path)
        self.model.eval()
        for mode in ["test", "train"]:
            testloader = cast(
                DataLoader,
                self.dataloaders[mode],
            )
            features = []
            labels = []

            with torch.inference_mode():
                for batch_idx, batch in enumerate(testloader):  # noqa: B007
                    input: torch.Tensor
                    target: torch.Tensor
                    input, target = batch
                    input = input[target < self.num_classes_in_selected_dataset]
                    target = target[target < self.num_classes_in_selected_dataset]
                    batch = (input, target)
                    if len(input) > 0:
                        (
                            feat,
                            label,
                        ) = self.extract_features_from_one_batch_for_caching_dataset(
                            batch=batch, mode=mode, batch_idx=batch_idx
                        )
                        features.append(feat)
                        labels.append(label)
                    if (batch_idx + 1) % 100 == 0:
                        features_tensor = torch.cat(features, dim=0).to("cpu")
                        torch.save(
                            features_tensor,
                            f"{path}/{mode}_features_{(batch_idx+1)//100}.pt",
                        )
                        labels_tensor = (
                            torch.cat(labels, dim=0)
                            .squeeze(2)
                            .to("cpu")
                            .type(torch.ByteTensor)
                        )
                        torch.save(
                            labels_tensor,
                            f"{path}/{mode}_labels_{(batch_idx+1)//100}.pt",
                        )
                        features = []
                        labels = []
                        del features_tensor
                        del labels_tensor

    def compute_metrics_for_batch_without_share_hidden(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        start_time = time()
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = self.metadata[mode]
        inputs = inputs[targets < self.num_classes_in_selected_dataset]
        targets = targets[targets < self.num_classes_in_selected_dataset]
        loss, loss_to_backprop, num_correct = self.model(
            x=inputs, y=targets, metadata=metadata
        )
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss_to_backprop.backward()
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

        average_accuracy_for_selected_paths = num_correct.sum() / total
        current_metric["average_accuracy_for_selected_paths"] = (
            average_accuracy_for_selected_paths,
            total,
        )
        average_loss_for_selected_paths = loss
        current_metric["average_loss_for_selected_paths"] = (
            average_loss_for_selected_paths,
            total,
        )

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

    @torch.inference_mode()
    def test_using_one_dataloader(self) -> None:
        epoch_start_time = time()
        self.model.eval()
        mode = "test"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        testloader = cast(
            DataLoader,
            self.dataloaders[mode],
        )
        encoder_fmodel, encoder_params, encoder_buffers = combine_state_for_ensemble(
            self.model.encoders
        )

        for batch_idx, batch in enumerate(testloader):  # noqa: B007
            input: torch.Tensor
            target: torch.Tensor
            input, target = batch
            if len(target.shape) == 1:
                input = input[target < self.num_classes_in_selected_dataset]
                target = target[target < self.num_classes_in_selected_dataset]
            batch = (input, target)
            if len(input) > 0:
                current_metric = self.compute_metrics_for_batch_for_eval(
                    encoder_fmodel=encoder_fmodel,
                    encoder_params=encoder_params,
                    encoder_buffers=encoder_buffers,
                    batch=batch,
                    mode=mode,
                    batch_idx=batch_idx,
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
