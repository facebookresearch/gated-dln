# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Collection of utility functions"""

import datetime
import gzip
import io
import os
import pathlib
import random
import re
import shutil
import subprocess  # noqa: S404
from typing import Any, Iterator, List, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")


def flatten_list(_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list

    Args:
        _list (List[List[Any]]): List of lists

    Returns:
        List[Any]: Flattened list
    """
    return [item for sublist in _list for item in sublist]


def chunks(_list: List[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from given list.
    Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    Args:
        _list (List[T]): list to chunk.
        n (int): size of chunks.

    Yields:
        Iterator[List[T]]: iterable over the chunks
    """
    for index in range(0, len(_list), n):
        yield _list[index : index + n]  # noqa: E203


def make_dir(path: str) -> pathlib.Path:
    """Make a directory, along with parent directories.
    Does not return an error if the directory already exists.

    Args:
        path (str): path to make the directory.

    Returns:
        pathlib.Path: path of the new directory.
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_current_commit_id() -> str:
    """Get current commit id.

    Returns:
        str: current commit id.
    """
    command = "git rev-parse HEAD"
    commit_id = (
        subprocess.check_output(command.split()).strip().decode("utf-8")  # noqa: S603
    )
    return commit_id


def has_uncommitted_changes() -> bool:
    """Check if there are uncommited changes.

    Returns:
        bool: wether there are uncommiteed changes.
    """
    command = "git status"
    output = subprocess.check_output(command.split()).strip().decode("utf-8")
    return "nothing to commit (working directory clean)" not in output


def set_seed(seed: int) -> None:
    """Set the seed for python, numpy, and torch.

    Args:
        seed (int): seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def unmerge_first_and_second_dim(
    batch: torch.Tensor, first_dim: int = -1, second_dim: int = -1
) -> torch.Tensor:
    """Modify the shape of a batch by unmerging the first dimension.
    Given a tensor of shape (a*b, c, ...), return a tensor of shape (a, b, c, ...).

    Args:
        batch (torch.Tensor): input batch.
        first_dim (int, optional): first dim. Defaults to -1.
        second_dim (int, optional): second dim. Defaults to -1.

    Returns:
        torch.Tensor: modified batch.
    """
    shape = batch.shape
    return batch.view(first_dim, second_dim, *shape[1:])


def split_on_caps(input_str: str) -> List[str]:
    """Split a given string at uppercase characters.
    Taken from: https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters

    Args:
        input_str (str): string to split.

    Returns:
        List[str]: splits of the given string.
    """
    return re.findall("[A-Z][^A-Z]*", input_str)


def is_integer(n: Union[int, str, float]) -> bool:
    """Check if the given value can be interpreted as an integer.

    Args:
        n (Union[int, str, float]): value to check.

    Returns:
        bool: can be the value be interpreted as an integer.
    """
    try:
        int(n)
    except ValueError:
        return False
    else:
        return True


def compress_file(
    path: str, format: str = "gzip", should_delete_uncompressed_file: bool = True
) -> str:
    new_path = f"{path}.gz"
    with open(path, "rb") as fin:
        with gzip.open(new_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    print(f"Copied replay buffer at {path} to {new_path}")
    if should_delete_uncompressed_file:
        os.remove(path)
        print(f"Deleted replay buffer at {path}")
    return new_path


def load_tensor(path: str) -> torch.Tensor:
    if path.endswith(".gz"):
        return read_compressed_tensor(path=path)
    else:
        return torch.load(path)


def read_compressed_tensor(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        file_content = f.read()
    return torch.load(io.BytesIO(file_content))


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_slurm_id() -> str:
    slurm_id = []
    env_var_names = ["SLURM_JOB_ID", "SLURM_STEP_ID"]
    for var_name in env_var_names:
        if var_name in os.environ:
            slurm_id.append(str(os.environ[var_name]))
    if slurm_id:
        return "-".join(slurm_id)
    return "-1"
