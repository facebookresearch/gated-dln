"""This is the main entry point for the code."""


import hydra
from omegaconf import DictConfig, OmegaConf

from src.app.run import run
from src.config_builder.register import prepare_for_loading_configs

config_path = "config"
config_name = "k_path_model"

prepare_for_loading_configs()


@hydra.main(config_path=config_path, config_name=config_name)
def launch(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config, resolve=True))
    return run(config)


if __name__ == "__main__":
    launch()
