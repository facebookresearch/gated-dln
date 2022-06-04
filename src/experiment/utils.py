import shutil
import time
from typing import List

import hydra
import torch
from omegaconf import DictConfig
from xplogger.logbook import LogBook

from src.experiment.base import Experiment
from src.utils import config as config_utils
from src.utils.utils import set_seed


def prepare(config: DictConfig, logbook: LogBook) -> Experiment:
    """Prepare an experiment

    Args:
        config (DictConfig): config of the experiment
        logbook (LogBook):
    """

    set_seed(seed=config.setup.seed)
    logbook.write_message(
        f"Starting Experiment at {time.asctime(time.localtime(time.time()))}"
    )
    logbook.write_message(f"torch version = {torch.__version__}")

    experiment = hydra.utils.instantiate(
        config.experiment.builder, config, logbook
    )  # cant seem to pass as a kwargs
    return experiment


def prepare_and_run(config: DictConfig, logbook: LogBook) -> None:
    """Prepare an experiment and run the experiment

    Args:
        config (DictConfig): config of the experiment
        logbook (LogBook):
    """
    experiment = prepare(config=config, logbook=logbook)
    experiment.run()


def prepare_and_extract_features(config: DictConfig, logbook: LogBook) -> None:
    """Prepare an experiment and extract features

    Args:
        config (DictConfig): config of the experiment
        logbook (LogBook):
    """

    set_seed(seed=config.setup.seed)
    logbook.write_message(
        f"Starting Experiment at {time.asctime(time.localtime(time.time()))}"
    )
    logbook.write_message(f"torch version = {torch.__version__}")
    experiment = hydra.utils.instantiate(
        config.experiment.builder, config, logbook
    )  # cant seem to pass as a kwargs
    experiment.extract_features_for_caching_dataset()


def clear(config: DictConfig) -> None:
    """Clear an experiment and delete all its data/metadata/logs
    given a config

    Args:
        config (DictConfig): config of the experiment to be cleared
    """

    for dir_to_del in get_dirs_to_delete_from_experiment(config):
        shutil.rmtree(dir_to_del)


def get_dirs_to_delete_from_experiment(config: DictConfig) -> List[str]:
    """Return a list of dirs that should be deleted when clearing an
        experiment

    Args:
        config (DictConfig): config of the experiment to be cleared

    Returns:
        List[str]: List of directories to be deleted
    """
    return [config.setup.save_dir]
