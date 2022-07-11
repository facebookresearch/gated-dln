# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from xplogger.logbook import LogBook

from src.utils.types import OptimizerType
from src.utils.utils import is_integer

ModelOrOptimizerType = Union[nn.Module, OptimizerType]


def save_metadata(save_dir: str, step: int, logbook: LogBook) -> None:
    """Save the metadata.

    Args:
        save_dir (str): directory to save.
        step (int): step for tracking the training of the agent.

    """
    metadata = {"step": step}
    path_to_save_at = f"{save_dir}/metadata.pt"
    torch.save(metadata, path_to_save_at)
    logbook.write_message(f"Saved {path_to_save_at}")


def save_random_state(save_dir: str, step: int, logbook: LogBook) -> None:
    """Save the metadata.

    Args:
        save_dir (str): directory to save.
        step (int): step for tracking the training of the agent.

    """
    random_state = {
        "np": np.random.get_state(),
        "python": random.getstate(),
        "pytorch": torch.get_rng_state(),
    }
    path_to_save_at = f"{save_dir}/random_state.pt"
    torch.save(random_state, path_to_save_at)
    logbook.write_message(f"Saved {path_to_save_at}")


def save_gate(
    gate: torch.Tensor,
    save_dir_path: pathlib.Path,
    logbook: LogBook,
) -> None:
    """Save the model.

    Args:
        model (nn.Module):
        name (str):
        save_dir_path (pathlib.Path): directory to save.
        step (int): step for tracking the training of the agent.
        retain_last_n (int): number of models to retain.
    """
    path_to_save_at = f"{save_dir_path}/gate.pt"
    torch.save(gate, path_to_save_at)
    logbook.write_message(f"Saved {path_to_save_at}")


def _save_model_or_optimizer(
    model_or_optimizer: ModelOrOptimizerType,
    name: str,
    save_dir_path: pathlib.Path,
    step: int,
    retain_last_n: int,
    logbook: LogBook,
) -> None:
    """Save the model_or_optimizer.

    Args:
        model_or_optimizer (ModelOrOptimizerType):
        name (str):
        save_dir_path (pathlib.Path): directory to save.
        step (int): step for tracking the training of the agent.
        retain_last_n (int): number of models to retain.
        suffix (str, optional): suffix to add at the name of the model before
            checkpointing. Defaults to "".
    """

    if model_or_optimizer is not None:
        path_to_save_at = f"{save_dir_path}/{name}_{step}.pt"
        torch.save(model_or_optimizer.state_dict(), path_to_save_at)
        logbook.write_message(f"Saved {path_to_save_at}")
        if retain_last_n == -1:
            return
        reverse_sorted_existing_versions = _get_reverse_sorted_existing_versions(
            save_dir_path, name
        )
        if len(reverse_sorted_existing_versions) > retain_last_n:
            # assert len(reverse_sorted_existing_versions) == retain_last_n + 1
            for path_to_del in reverse_sorted_existing_versions[retain_last_n:]:
                if os.path.lexists(path_to_del):
                    os.remove(path_to_del)
                    logbook.write_message(f"Deleted {path_to_del}")


def save_model(
    model: nn.Module,
    name: str,
    save_dir_path: pathlib.Path,
    step: int,
    retain_last_n: int,
    logbook: LogBook,
) -> None:
    """Save the model.

    Args:
        model (nn.Module):
        name (str):
        save_dir_path (pathlib.Path): directory to save.
        step (int): step for tracking the training of the agent.
        retain_last_n (int): number of models to retain.
    """
    return _save_model_or_optimizer(
        model_or_optimizer=model,
        name=name,
        save_dir_path=save_dir_path,
        step=step,
        retain_last_n=retain_last_n,
        logbook=logbook,
    )


def save_optimizer(
    optimizer: OptimizerType,
    name: str,
    save_dir_path: pathlib.Path,
    step: int,
    retain_last_n: int,
    logbook: LogBook,
) -> None:
    """Save the optimizer.

    Args:
        optimizer (OptimizerType):
        name (str):
        save_dir_path (pathlib.Path): directory to save.
        step (int): step for tracking the training of the agent.
        retain_last_n (int): number of models to retain.
    """
    return _save_model_or_optimizer(
        model_or_optimizer=optimizer,
        name=name,
        save_dir_path=save_dir_path,
        step=step,
        retain_last_n=retain_last_n,
        logbook=logbook,
    )


def _get_reverse_sorted_existing_versions(
    save_dir_path: pathlib.Path, name: str
) -> List[str]:
    """List of existing checkpoints in reverse sorted order.

    Args:
        save_dir_path (Path): directory to find checkpoints in.
        name (str): name of the checkpoint.

    Returns:
        List[str]: list of checkpoints in reverse sorted order.
    """
    existing_versions: List[str] = [str(x) for x in save_dir_path.glob(f"{name}_*.pt")]
    existing_versions = [
        x
        for x in existing_versions
        if is_integer(x.rsplit("/", 1)[-1].replace(f"{name}_", "").replace(".pt", ""))
    ]
    existing_versions.sort(reverse=True, key=_get_step_from_checkpoint_path)
    return existing_versions


def _get_step_from_checkpoint_path(_path: str) -> int:
    """Parse the checkpoint path to obtain the step

    Args:
        _path (str): path to the checkpoint.

    Returns:
        int: step for tracking the training of the agent.
    """
    return int(_path.rsplit("/", 1)[-1].replace(".pt", "").rsplit("_", 1)[-1])


def _load_model_or_optimizer(
    model_or_optimizer: ModelOrOptimizerType,
    save_dir: str,
    name: str,
    step: int,
    logbook: LogBook,
) -> ModelOrOptimizerType:
    """Load a model.

    Args:
        model_or_optimizer (ModelOrOptimizerType): model_or_optimizer to load.
        save_dir (str): directory to load from.
        name (str): name of the model_or_optimizer.
        step (int): step for tracking the training of the agent.

    Returns:
        ModelOrOptimizerType: loaded model or
            optimizer.
    """

    assert model_or_optimizer is not None
    path_to_load_from = f"{save_dir}/{name}_{step}.pt"
    logbook.write_message(f"Path to load from: {path_to_load_from}")
    if os.path.exists(path_to_load_from):
        model_or_optimizer.load_state_dict(torch.load(path_to_load_from))
        logbook.write_message(f"Loading from path: {path_to_load_from}")
    else:
        logbook.write_message(f"No model_or_optimizer to load from {path_to_load_from}")
    return model_or_optimizer


def load_model(
    model: nn.Module, save_dir: str, name: str, step: int, logbook: LogBook
) -> nn.Module:
    """Load a model.

    Args:
        model (nn.Module): model_or_optimizer to load.
        save_dir (str): directory to load from.
        name (str): name of the model_or_optimizer.
        step (int): step for tracking the training of the agent.

    Returns:
        model: loaded model
    """
    return _load_model_or_optimizer(  # type: ignore[return-value]
        model_or_optimizer=model,
        save_dir=save_dir,
        name=name,
        step=step,
        logbook=logbook,
    )
    # mypy error: Incompatible return value type (got "Union[Module, Optimizer]", expected Module)  [return-value]


def load_optimizer(
    optimizer: OptimizerType, save_dir: str, name: str, step: int, logbook: LogBook
) -> OptimizerType:
    """Load an optimizer.

    Args:
        optimizer (OptimizerType): model_or_optimizer to load.
        save_dir (str): directory to load from.
        name (str): name of the model_or_optimizer.
        step (int): step for tracking the training of the agent.

    Returns:
        optimizer: loaded optimizer
    """
    return _load_model_or_optimizer(  # type: ignore[return-value]
        model_or_optimizer=optimizer,
        save_dir=save_dir,
        name=name,
        step=step,
        logbook=logbook,
    )
    # mypy error: Incompatible return value type (got "Union[Module, Optimizer]", expected Module)  [return-value]


def load_metadata(save_dir: str, logbook: LogBook) -> Optional[Dict[Any, Any]]:
    """Load the metadata.

    Args:
        save_dir (str): directory to load the model from.

    Returns:
        Optional[Dict[Any, Any]]: metadata.
    """
    metadata_path = f"{save_dir}/metadata.pt"
    if not os.path.exists(metadata_path):
        logbook.write_message(f"{metadata_path} does not exist.")
        metadata = None
    else:
        metadata = torch.load(metadata_path)
    return metadata


def load_gate(save_dir: str, logbook: LogBook) -> torch.Tensor:
    """Load the gate.

    Args:
        save_dir (str): directory to load the model from.

    Returns:
        Optional[Dict[Any, Any]]: metadata.
    """
    gate_path = f"{save_dir}/gate.pt"
    if not os.path.exists(gate_path):
        logbook.write_message(f"{gate_path} does not exist.")
        raise RuntimeError(f"{gate_path} does not exist.")
    else:
        gate = torch.load(gate_path)
    return gate


def get_random_state(
    save_dir: str, step: int, logbook: LogBook
) -> Optional[Dict[str, Any]]:
    """Get the random state.

    Args:
        save_dir (str): directory to load the model from.

    Returns:
        Optional[Dict[Any, Any]]: random_state.
    """

    random_state_path = f"{save_dir}/random_state.pt"
    if not os.path.exists(random_state_path):
        logbook.write_message(f"{random_state_path} does not exist.")
        random_state = None
    else:
        random_state = torch.load(random_state_path)
    return random_state


def load_random_state(save_dir: str, step: int, logbook: LogBook) -> None:
    """Load the random state for the model.

    Args:
        save_dir (str): directory to load the model from.

    """
    random_state = get_random_state(save_dir=save_dir, step=step, logbook=logbook)
    assert random_state is not None
    np.random.set_state(random_state["np"])
    random.setstate(random_state["python"])
    torch.set_rng_state(random_state["pytorch"])
