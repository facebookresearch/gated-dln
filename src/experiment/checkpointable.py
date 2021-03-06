# Copyright (c) Meta Platforms, Inc. and affiliates.
import pathlib
from typing import Union

import torch
import torch.utils.data
from omegaconf.dictconfig import DictConfig
from xplogger.logbook import LogBook

from src.checkpoint import utils as checkpoint_utils
from src.data.utils import get_num_classes_from_dataloader_name
from src.ds.experiment import ExperimentMetadata, TrainState
from src.model.base import Model as BaseModel
from src.utils import utils
from src.utils.types import OptimizerType


class Experiment:
    """Checkpointable Experiment Class"""

    def __init__(self, cfg: DictConfig, logbook: LogBook, experiment_id: str = "0"):
        """Experiment Class

        Args:
            cfg (cfgType):
            logbook (LogBook):
            experiment_id (str, optional): Defaults to "0".
        """
        self.id = experiment_id
        self.cfg = cfg
        self.validate_cfg()
        self.logbook = logbook
        self.device = torch.device(self.cfg.setup.device)
        self.dataloaders: Union[
            dict[str, torch.utils.data.DataLoader],
            dict[str, tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
        ]

        self.model: BaseModel

        self.save_dir = utils.make_dir(self.cfg.experiment.save.dir)
        self.optimizer: OptimizerType
        self.train_state: TrainState
        self._supported_modes: list[str]
        self.should_write_batch_logs: bool
        self.metadata: dict[str, ExperimentMetadata]

    def validate_cfg(self):
        if self.cfg.model.name in [
            "main",
        ]:
            assert not self.cfg.model.decoder_cfg.should_share
            if self.cfg.model.name in ["k_path_model_without_share_hidden"]:
                assert not self.cfg.model.hidden_layer_cfg.should_share
            else:
                assert self.cfg.model.hidden_layer_cfg.should_share
        if self.cfg.dataloader.name == "cifar10_v1":
            assert self.cfg.experiment.task_one.name == "living_or_not"
            assert self.cfg.experiment.task_two.name == "fly_or_not"
        if (
            self.cfg.dataloader._target_
            == "src.data.filesystem.dataloader.build_dataloaders"
        ):
            assert (
                self.cfg.experiment.task.num_classes_in_selected_dataset
                == get_num_classes_from_dataloader_name(name=self.cfg.dataloader.name)
            )
        if not self.cfg.setup.id.startswith("preprocessed_"):
            assert self.cfg.model.pretrained_cfg.should_use is False
        if self.cfg.dataloader.name == "mnist":
            assert self.cfg.dataloader.is_preprocessed is False

    def save(self, step: int) -> None:
        """Method to save the experiment"""

        retain_last_n = self.cfg.experiment.save.retain_last_n
        if retain_last_n == 0:
            self.logbook.write_message("Not saving the experiment as retain_last_n = 0")
            return

        # save_dir_path = pathlib.Path(self.save_dir)

        self.model.save(
            name="model",
            save_dir_path=self.save_dir,
            step=step,
            retain_last_n=retain_last_n,
            logbook=self.logbook,
        )

        self.save_optimizer(
            save_dir_path=self.save_dir,
            step=step,
            retain_last_n=retain_last_n,
        )

        self.save_metadata(step)
        self.save_random_state(step)

    def save_optimizer(
        self,
        save_dir_path: pathlib.Path,
        step: int,
        retain_last_n: int,
    ) -> None:
        """Save the optimizer.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.

        """

        return checkpoint_utils.save_optimizer(
            optimizer=self.optimizer,
            name="optimizer",
            save_dir_path=save_dir_path,
            step=step,
            retain_last_n=retain_last_n,
            logbook=self.logbook,
        )

    def save_metadata(self, step: int) -> None:
        """Save the metadata.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.

        """
        return checkpoint_utils.save_metadata(
            save_dir=self.save_dir, step=step, logbook=self.logbook
        )

    def save_random_state(self, step: int) -> None:
        """Save the random_state.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.

        """
        return checkpoint_utils.save_random_state(
            save_dir=self.save_dir, step=step, logbook=self.logbook
        )

    def load_latest_step(self) -> int:
        """Load the agent using the latest training step.

        Args:
            model_dir (Optional[str]): directory to load the model from.

        Returns:
            int: step for tracking the training of the agent.
        """
        latest_step = -1
        metadata = checkpoint_utils.load_metadata(
            save_dir=self.save_dir, logbook=self.logbook
        )
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        self.load(step=latest_step)
        if latest_step == 0:
            # special case when no step has been taken so far.
            return latest_step
        return latest_step + 1

    def load(self, step: int) -> None:
        """Method to load the entire experiment"""

        self.model.load(
            name="model", save_dir=self.save_dir, step=step, logbook=self.logbook
        )
        self.load_optimizer(step=step)

        checkpoint_utils.load_random_state(
            save_dir=self.save_dir, step=step, logbook=self.logbook
        )

    def load_optimizer(self, step: int) -> None:
        """Method to load the optimizer"""
        checkpoint_utils.load_optimizer(
            optimizer=self.optimizer,
            save_dir=self.save_dir,
            step=step,
            name="optimizer",
            logbook=self.logbook,
        )

    def periodic_save(self, train_state: TrainState) -> None:
        """Perioridically save the experiment.

        This is a utility method, built on top of the `save` method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch.
        Args:
            step (int): current step.
        """
        save_frequency = self.cfg.experiment.save.frequency
        if save_frequency > 0 and train_state.epoch % save_frequency == 0:
            self.save(train_state.step)
