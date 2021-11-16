from __future__ import annotations

import pytest

from tests.data_preprocessing.utils import check_output_from_cmd
from tests.utils import get_overrides_to_test


def get_overrides_to_test_cifar10_data_preprocessing():

    params_to_test = {
        "experiment.task.num_input_transformations": [5],
        "experiment.task.num_classes_in_selected_dataset": [
            10,
        ],
        "dataloader.train_config.dataloader.batch_size": [
            100,
        ],
        "dataloader": ["cifar10"],
        "experiment": ["k_path_model"],
        "experiment.task.mode": ["rotate_input_permute_target"],
        "model.gate_cfg.mode": ["8_plus_mod"],
    }

    return get_overrides_to_test(params_to_test=params_to_test)


@pytest.mark.parametrize(
    "overrides",
    get_overrides_to_test_cifar10_data_preprocessing(),
)
def test_cifar10_data_preprocessing(overrides) -> None:
    check_output_from_cmd(overrides=overrides)
