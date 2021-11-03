# type: ignore
"""Class to interface with an Experiment"""

from __future__ import annotations

from time import time

import torch
import torch.utils.data
from xplogger.logbook import LogBook
from xplogger.types import LogType

from src.experiment import base as base_experiment
from src.experiment.ds import ExperimentMode, Metadata, MoeMaskMode
from src.utils.config import DictConfig


class Experiment(base_experiment.Experiment):
    """Experiment Class"""

    def __init__(self, cfg: DictConfig, logbook: LogBook, experiment_id: str = "0"):
        """Experiment Class

        Args:
            config (DictConfig):
            logbook (LogBook):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(cfg=cfg, logbook=logbook, experiment_id=experiment_id)
        self.loss_fn = None

    def compute_metrics_for_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        start_time = time()
        should_train = mode == "train"
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = Metadata(
            experiment_mode=ExperimentMode(mode),
            moe_mask_mode=MoeMaskMode(self.cfg.experiment.moe.mask.mode[mode]),
        )
        outputs, loss, num_correct = self.model(x=inputs, y=targets, metadata=metadata)
        if should_train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        total = targets.size(0)

        current_metric = {
            "loss": loss.item(),
            "accuracy": num_correct * 100.0 / total,
            "batch_index": batch_idx,
            "epoch": self.train_state.epoch,
            "step": self.train_state.step,
            "time_taken": time() - start_time,
            "mode": mode,
        }
        if self.should_write_batch_logs:
            self.logbook.write_metric(metric=current_metric)
        current_metric.pop("time_taken")
        return current_metric
