"""This is the main entry point for the running the experiments."""

import hydra
from notifiers import get_notifier
from omegaconf import DictConfig
from xplogger.logbook import LogBook

from src.experiment import utils as experiment_utils
from src.utils import config as config_utils


def run(config: DictConfig) -> None:
    """Create and run the experiment.

    Args:
        config (DictConfig): config for the experiment.
    """
    config_utils.pretty_print(config, resolve=False)
    config_id = config.setup.id
    logbook_config = config_utils.to_dict(config.logbook, resolve=True)
    logbook_config.pop("should_write_batch_logs")
    logbook_config = hydra.utils.instantiate(logbook_config)
    if "mongo" in logbook_config["loggers"] and (
        config_id.startswith("pytest_")
        or config_id in ["sample", "sample_config"]
        or config_id.startswith("test_")
        # or is_debug_job
    ):
        # do not write the job to mongo db.
        print(logbook_config["loggers"].pop("mongo"))
    logbook = LogBook(logbook_config)

    experiment_utils.prepare_and_extract_features(config=config, logbook=logbook)
