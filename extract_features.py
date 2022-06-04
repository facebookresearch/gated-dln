"""This is the main entry point for the code."""


import hydra
from oc_extras.resolvers import register_new_resolvers
from omegaconf import DictConfig, OmegaConf

from src.app.extract_features import run

config_path = "config"
config_name = "main"

register_new_resolvers()


@hydra.main(config_path=config_path, config_name=config_name)
def launch(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config, resolve=True))
    return run(config)


if __name__ == "__main__":
    launch()
