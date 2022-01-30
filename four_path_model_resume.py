"""This is the main entry point for the code."""


import hydra
from omegaconf import DictConfig

from src.app.run import run
from src.config_builder.register import prepare_for_loading_configs
from src.utils import config as config_utils

config_path = "config"
config_name = "four_path_model"

prepare_for_loading_configs()


@hydra.main(config_path=config_path, config_name=config_name)
def launch(config: DictConfig) -> None:
    config = config_utils.read_config_from_file_for_resuming(
        config_path=f"{config.setup.base_path}/logs/{config.setup.id}/config_log.jsonl"
    )
    config = config_utils.make_config_mutable(config)
    return run(config)


if __name__ == "__main__":
    launch()
