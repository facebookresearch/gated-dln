from __future__ import annotations

import subprocess
import sys
from typing import Dict

OverridesType = Dict[str, str]


def get_command(overrides: OverridesType) -> list[str]:
    parameters = {
        "setup.seed": 1,
        "setup.description": "pytest",
        "setup.id": get_id_from_overrides(overrides),
    }
    parameters.update(overrides)
    cmd = [sys.executable, "extract_features.py"]
    for key, value in parameters.items():
        if "$" in str(key) or "$" in str(value):
            cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={value}")
    return cmd


def check_output_from_cmd(**kwargs):
    cmd = get_command(**kwargs)
    print(" ".join(cmd))
    result = subprocess.check_output(cmd)
    actual_logs = set(result.decode("utf-8").split("\n"))
    return len(actual_logs) > 0


# def map_config_to_string(config: Dict[str, Any]) -> str:
#     config_str = "_".join(
#         [
#             f"{key}-{value}"
#             for (key, value) in config.items()
#             if "$" not in str(key) and "$" not in str(value)
#         ]
#     )
#     return hashlib.sha224(config_str.encode()).hexdigest()


def get_id_from_overrides(overrides: OverridesType) -> str:
    return f"preprocessed_{overrides['dataloader']}_dataset_{overrides['experiment.task.num_classes_in_selected_dataset']}_classes_input_rotated_output_permuted_{overrides['experiment.task.num_input_transformations']}_transformations_v1"
