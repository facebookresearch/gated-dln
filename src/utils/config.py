"""Code to interface with the config."""
from __future__ import annotations

import importlib
import json
from copy import deepcopy
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def instantiate_using_config(cfg: DictConfig, *args, **kwargs):
    # Special cases for instantiating optimizers
    # To be removed when we switch to Hydra 1.1
    module_name, cls_name = cfg._target_.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    cfg_to_init = OmegaConf.to_container(cfg=cfg, resolve=True)
    cfg_to_init.pop("_target_")
    cfg_to_init.update(kwargs)
    return cls(*args, **cfg_to_init)


def get_empty_config() -> DictConfig:
    return OmegaConf.create({})


def dict_to_config(dictionary: dict) -> DictConfig:
    """Convert the dictionary to a config.

    Args:
        dictionary (Dict): dictionary to convert.

    Returns:
        DictConfig: config made from the dictionary.
    """
    return OmegaConf.create(dictionary)


def make_config_mutable(config: DictConfig) -> DictConfig:
    """Set the config to be mutable.

    Args:
        config (DictConfig):

    Returns:
        DictConfig:
    """
    OmegaConf.set_readonly(config, False)
    return config


def make_config_immutable(config: DictConfig) -> DictConfig:
    """Set the config to be immutable.

    Args:
        config (DictConfig):

    Returns:
        DictConfig:
    """
    OmegaConf.set_readonly(config, True)
    return config


def set_struct(config: DictConfig) -> DictConfig:
    """Set the struct flag in the config.

    Args:
        config (DictConfig):

    Returns:
        DictConfig:
    """
    OmegaConf.set_struct(config, True)
    return config


def unset_struct(config: DictConfig) -> DictConfig:
    """Unset the struct flag in the config.

    Args:
        config (DictConfig):

    Returns:
        DictConfig:
    """
    OmegaConf.set_struct(config, False)
    return config


def to_dict(config: DictConfig, resolve: bool = False) -> dict[str, Any]:
    """Convert config to a dictionary.

    Args:
        config (DictConfig):

    Returns:
        Dict:
    """
    dict_config = cast(
        dict[str, Any], OmegaConf.to_container(deepcopy(config), resolve=resolve)
    )
    return dict_config


def process_config(config: DictConfig, should_make_dir: bool = True) -> DictConfig:
    """Process the config.

    Args:
        config (DictConfig): config object to process.
        should_make_dir (bool, optional): should make dir for saving logs, models etc? Defaults to True.

    Returns:
        DictConfig: processed config.
    """
    return set_struct(make_config_immutable(config))


def read_config_from_file(config_path: str) -> DictConfig:
    """Read the config from filesystem.

    Args:
        config_path (str): path to read config from.

    Returns:
        DictConfig:
    """
    config = OmegaConf.load(config_path)
    assert isinstance(config, DictConfig)
    return set_struct(make_config_immutable(config))


def read_config_from_file_for_resuming(config_path: str) -> DictConfig:
    """Read the config from filesystem.

    Args:
        config_path (str): path to read config from.

    Returns:
        DictConfig:
    """
    with open(config_path) as f:
        for line in f:
            config = OmegaConf.create(json.loads(line))
            break
    assert isinstance(config, DictConfig)
    return set_struct(make_config_immutable(config))


def pretty_print(config, resolve: bool = True):
    """Prettyprint the config.

    Args:
        config ([type]):
        resolve (bool, optional): should resolve the config before printing. Defaults to True.
    """
    print(OmegaConf.to_yaml(config, resolve=resolve))
