"""Class to interface with an Experiment"""

from __future__ import annotations

import json
from time import time
from typing import Union

import hydra
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from xplogger import metrics as ml_metrics
from xplogger.logbook import LogBook
from xplogger.types import LogType

from src.data.batch_list import BatchList
from src.experiment import checkpointable as checkpointable_experiment
from src.experiment.ds import ExperimentMetadata, ExperimentMode, TrainState
from src.model.base import Model as BaseModel
from src.utils import config as config_utils
from src.utils.types import OptimizerType


class Experiment(checkpointable_experiment.Experiment):
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
        super().__init__(cfg=cfg, logbook=logbook, experiment_id=experiment_id)
        if cfg.dataloader._target_.endswith("build_task_specific_dataloaders"):
            self.should_use_task_specific_dataloaders = True
        else:
            self.should_use_task_specific_dataloaders = False
        if self.should_use_task_specific_dataloaders:
            self.train = self.train_using_two_dataloaders
            self.test = self.test_using_two_dataloaders
        else:
            if self.use_preprocessed_dataset:
                self.train = (
                    self.train_using_one_dataloader_when_using_preprocessed_dataset
                )
            else:
                self.train = (
                    self.train_using_one_dataloader_when_using_unprocessed_dataset
                )
            self.test = self.test_using_one_dataloader
        self.dataloaders: Union[
            dict[str, torch.utils.data.DataLoader],
            dict[str, tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
        ]

        if should_init:
            self.dataloaders = hydra.utils.instantiate(self.cfg.dataloader)
            self.model: BaseModel = hydra.utils.instantiate(self.cfg.model).to(
                self.device
            )
            assert isinstance(self.model, BaseModel)
            self.optimizer: OptimizerType = hydra.utils.instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )
            self._supported_modes: list[str] = ["train", "test"]
            self.metadata = {
                mode: ExperimentMetadata(mode=ExperimentMode(mode))
                for mode in self._supported_modes
            }

            self._post_init()

    def _post_init(self) -> None:

        assert all(key in self.dataloaders for key in self._supported_modes)
        start_step = 0

        should_resume_experiment = self.cfg.experiment.should_resume

        if should_resume_experiment:
            start_step = self.load_latest_step()

        self.train_state = self._make_train_state(start_step=start_step)

        self.should_write_batch_logs = self.cfg.logbook.should_write_batch_logs
        self.startup_logs()

    def _make_train_state(self, start_step: int) -> TrainState:
        if self.should_use_task_specific_dataloaders:
            assert isinstance(self.dataloaders["train"], tuple)
            train_state = TrainState(
                num_batches_per_epoch=len(self.dataloaders["train"][0]), step=start_step
            )
        else:
            train_state = TrainState(
                num_batches_per_epoch=len(self.dataloaders["train"]), step=start_step
            )

        return train_state

    def startup_logs(self) -> None:
        """Write some logs at the start of the experiment."""
        config_file = f"{self.cfg.setup.save_dir}/config.json"
        with open(config_file, "w") as f:
            f.write(json.dumps(config_utils.to_dict(self.cfg, resolve=True)))
        self.logbook.write_message(
            f"Number of parameters = {self.model.get_param_count()}, Number of trainable parameters = {self.model.get_trainable_param_count()}"
        )

    def run(self) -> None:
        start_epoch = self.train_state.epoch
        for _ in range(start_epoch, start_epoch + self.cfg.experiment.num_epochs):
            self.train()
            self.test()
            self.periodic_save(train_state=self.train_state)

    def train_using_one_dataloader(self) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        assert isinstance(self.dataloaders[mode], DataLoader)
        for batch in self.dataloaders[mode]:  # noqa: B007
            current_metric = self.compute_metrics_for_batch(
                batch=batch,  # type: ignore[arg-type]
                # Argument "batch" to "compute_metrics_for_batch" of "Experiment" has incompatible type "Union[Any, DataLoader[Any]]"; expected "Tuple[Tensor, Tensor]"  [arg-type]
                mode=mode,
                batch_idx=self.train_state.batch,
            )
            self.train_state.step += 1
            metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def train_using_two_dataloaders(self) -> None:
        epoch_start_time = time()
        self.model.train()
        mode = "train"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        dataloader1_iter = iter(self.dataloaders[mode][0])  # type: ignore[index]
        # Value of type "Union[DataLoader[Any], Tuple[DataLoader[Any], DataLoader[Any]]]" is not indexable  [index]
        for batch2 in self.dataloaders[mode][1]:  # type: ignore[index]
            # Value of type "Union[DataLoader[Any], Tuple[DataLoader[Any], DataLoader[Any]]]" is not indexable  [index]
            batch1 = next(dataloader1_iter)
            current_metric = self.compute_metrics_for_batch(
                batch=[BatchList([batch1[i], batch2[i]]) for i in range(len(batch1))],  # type: ignore[arg-type]
                mode=mode,
                batch_idx=self.train_state.batch,
            )
            # error: Argument "batch" has incompatible type "List[Any]"; expected "Tuple[Tensor, Tensor]"  [arg-type]
            self.train_state.step += 1
            metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def test_using_one_dataloader(self) -> None:
        epoch_start_time = time()
        self.model.eval()
        mode = "test"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        testloader = self.dataloaders[mode]
        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):  # noqa: B007
                current_metric = self.compute_metrics_for_batch(
                    batch=batch, mode=mode, batch_idx=batch_idx
                )
                metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def test_using_two_dataloaders(self) -> None:
        epoch_start_time = time()
        self.model.eval()
        mode = "test"
        metric_dict = self.init_metric_dict(epoch=self.train_state.epoch, mode=mode)
        testloader1_iter = iter(self.dataloaders[mode][0])  # type: ignore[index]
        # Value of type "Union[DataLoader[Any], Tuple[DataLoader[Any], DataLoader[Any]]]" is not indexable  [index]
        testloader2 = self.dataloaders[mode][1]  # type: ignore[index]
        # Value of type "Union[DataLoader[Any], Tuple[DataLoader[Any], DataLoader[Any]]]" is not indexable  [index]
        with torch.no_grad():
            for batch_idx, batch2 in enumerate(testloader2):  # noqa: B007
                batch1 = next(testloader1_iter)
                current_metric = self.compute_metrics_for_batch(
                    batch=[
                        BatchList([batch1[i], batch2[i]]) for i in range(len(batch1))  # type: ignore[arg-type]
                    ],
                    mode=mode,
                    batch_idx=batch_idx,
                )
                # error: Argument "batch" has incompatible type "List[Any]"; expected "Tuple[Tensor, Tensor]"  [arg-type]
                metric_dict.update(metrics_dict=current_metric)
        metric_dict = metric_dict.to_dict()
        metric_dict.pop("batch_index")
        metric_dict["time_taken"] = time() - epoch_start_time
        self.logbook.write_metric(metric=metric_dict)

    def compute_metrics_for_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        raise NotImplementedError

    def init_metric_dict(self, epoch: int, mode: str) -> ml_metrics.MetricDict:
        metric_dict = ml_metrics.MetricDict(
            [
                ml_metrics.AverageMetric("loss"),
                ml_metrics.AverageMetric("accuracy"),
                ml_metrics.ConstantMetric("epoch", epoch),
                ml_metrics.ConstantMetric("mode", mode),
                ml_metrics.CurrentMetric("batch_index"),
                ml_metrics.CurrentMetric("step"),
            ]
        )

        return metric_dict
