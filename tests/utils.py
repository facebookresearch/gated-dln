from __future__ import annotations

import itertools
from typing import Any

OverridesType = dict[str, str]


def map_combination_of_params_to_config(
    combination: list[Any], keys: list[str]
) -> dict[str, Any]:
    config = {}
    for index, key in enumerate(keys):
        config[key] = combination[index]
    return config


def get_overrides_to_test(params_to_test: dict[str, list[Any]]):
    combinations = list(itertools.product(*params_to_test.values()))
    configs_to_test = [
        map_combination_of_params_to_config(
            combination=combination, keys=list(params_to_test.keys())
        )
        for combination in combinations
    ]
    return configs_to_test
