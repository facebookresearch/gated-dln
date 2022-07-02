"""This is the main entry point for the running the experiments."""
from __future__ import annotations

import hydra
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
    is_debug_job = False
    slurm_id = config.setup.slurm_id
    if slurm_id == "-1":
        # the job is not running on slurm.
        is_debug_job = True
    config_id = config.setup.id
    logbook_config = config_utils.to_dict(config.logbook, resolve=True)
    logbook_config.pop("should_write_batch_logs")
    logbook_config = hydra.utils.instantiate(logbook_config)
    if "mongo" in logbook_config["loggers"] and (
        config_id.startswith("pytest_")
        or config_id in ["sample", "sample_config"]
        or config_id.startswith("test_")
        or config.setup.slurm.cloud != "fair"
        # or is_debug_job
    ):
        # do not write the job to mongo db.
        print(logbook_config["loggers"].pop("mongo"))
    logbook = LogBook(logbook_config)
    config_to_write = config_utils.to_dict(config, resolve=True)
    config_to_write["status"] = "RUNNING"
    logbook.write_metadata(config_to_write)
    logbook.write_config(config_to_write)

    experiment_utils.prepare_and_run(config=config, logbook=logbook)

    config_to_write["status"] = "COMPLETED"
    logbook.write_metadata(config_to_write)

    if not is_debug_job:
        zulip.notify(
            message=f"Completed experiment for config_id: {config_id}. Slurm id is {slurm_id}",
            **config.notifier,
        )
