"""This is the main entry point for the code."""

import hydra
from oc_extras.resolvers import register_new_resolvers
from omegaconf import DictConfig

from src.app.run import run
from src.utils import config as config_utils

config_path = "config"
config_name = "main"

register_new_resolvers()

@hydra.main(config_path=config_path, config_name=config_name)
def launch(config: DictConfig) -> None:
    try:
        config = config_utils.read_config_from_file_for_resuming(
            config_path=f"{config.setup.base_path}/logs/{config.setup.id}/config_log.jsonl"
        )
    except json.decoder.JSONDecodeError:
        config = config_utils.read_json_config_from_file(
            config_path=f"{config.setup.base_path}/logs/{config.setup.id}/config_log.jsonl"
        )

    config = config_utils.make_config_mutable(config)
    return run(config)


if __name__ == "__main__":
    launch()
