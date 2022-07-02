from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

OverridesType = Dict[str, str]


def get_command_and_id(overrides: OverridesType) -> list[str]:
    parameters = {
        "setup.seed": 1,
        "setup.description": "pytest",
    }
    parameters.update(overrides)
    if parameters["dataloader"] == "filesystem":
        parameters[
            "dataloader.name"
        ] = f"preprocessed_cifar10_dataset_{parameters['experiment.task.num_classes_in_selected_dataset']}_classes_{parameters['experiment.task.mode']}_{parameters['experiment.task.num_input_transformations']}_transformations_v1"

    setup_id = get_id_from_overrides(parameters)
    parameters["setup.id"] = setup_id

    cmd = [sys.executable, "main.py"]
    for key, value in parameters.items():
        if "$" in str(key) or "$" in str(value):
            cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={value}")
    return cmd, setup_id


def check_output_from_cmd(**kwargs):
    cmd, setup_id = get_command_and_id(**kwargs)
    print(" ".join(cmd))
    cwd = os.getcwd()
    log_path = f"{cwd}/logs/{setup_id}"
    assert os.path.exists(log_path) is False
    result = subprocess.check_output(cmd)
    actual_logs = set(result.decode("utf-8").split("\n"))
    shutil.rmtree(log_path)
    assert os.path.exists(log_path) is False
    return len(actual_logs) > 0


def map_config_to_string(config: dict[str, Any]) -> str:
    config_str = "_".join(
        [
            f"{key}-{value}"
            for (key, value) in config.items()
            if "$" not in str(key) and "$" not in str(value) and key != "setup.id"
        ]
    )
    return hashlib.sha224(config_str.encode()).hexdigest()


def get_id_from_overrides(overrides: OverridesType) -> str:
    return f"pytest_{map_config_to_string(overrides)}"
