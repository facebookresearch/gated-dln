"""Class to interface with an Experiment"""
from __future__ import annotations

import dataclasses
from typing import OrderedDict

import torch
import torch.utils.data
from omegaconf.dictconfig import DictConfig
from xplogger.logbook import LogBook
from xplogger.types import LogType

from src.experiment import k_path_model as base_experiment
from src.experiment.ds import ModelFeature


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
            should_init=True,
        )

        if self.cfg.model.hidden_layer_cfg.should_share:
            self.compute_features_for_batch = (
                self.compute_features_for_batch_when_using_shared_hidden_layer
            )
        else:
            self.compute_features_for_batch = (
                self.compute_features_for_batch_without_share_hidden
            )

    def compute_features_for_batch_when_using_shared_hidden_layer(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = self.metadata[mode]
        with torch.inference_mode():
            encoder_output, hidden_output = self.model.extract_features(
                x=inputs, y=targets, metadata=metadata
            )
        features = ModelFeature(
            encoder_output=torch.cat(encoder_output, dim=1).to("cpu"),
            hidden_output=hidden_output.view(
                batch[0].shape[0], self.model.gate.shape[1], hidden_output.shape[-1]
            ).to("cpu"),
            gate=self.model.gate.to("cpu"),
        )
        return features

    def compute_feature_using_one_dataloader(self) -> None:
        buffer = OrderedDict(
            {
                "input": torch.empty(0, 1, 28, 28),
                "target": torch.empty(0, dtype=torch.int64),
            }
        )
        mode = "train"
        for batch in self.dataloaders[mode]:  # noqa: B007
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
                model_feature = self.compute_features_for_batch(
                    batch=batch,
                    mode=mode,
                    batch_idx=self.train_state.batch,
                )
                torch.save(
                    dataclasses.asdict(model_feature),
                    f"{self.cfg.setup.save_dir}/model_features.tar",
                )
                print(f"Saved features at {self.cfg.setup.save_dir}/model_features.tar")
                return

    def compute_features_for_batch_without_share_hidden(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: str,
        batch_idx: int,
    ) -> LogType:
        inputs, targets = [_tensor.to(self.device) for _tensor in batch]
        metadata = self.metadata[mode]
        encoder_output, hidden_output = self.model.extract_features(
            x=inputs, y=targets, metadata=metadata
        )
        features = ModelFeature(
            encoder_output=encoder_output,
            hidden_output=hidden_output,
            gate=self.model.gate,
        )

        return features

    def run(self) -> None:
        self.compute_feature_using_one_dataloader()
