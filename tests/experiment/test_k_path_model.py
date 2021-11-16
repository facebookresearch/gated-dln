from __future__ import annotations

import pytest

from tests.experiment.utils import check_output_from_cmd
from tests.utils import get_overrides_to_test


def get_overrides_to_test_mnist():

    params_to_test = {
        "experiment.num_epochs": [2],
        "experiment.task.num_input_transformations": [5],
        "experiment.task.num_classes_in_selected_dataset": [6, 8],
        "dataloader.train_config.dataloader.batch_size": [
            100,
        ],
        "dataloader": ["mnist"],
        "experiment": ["k_path_model"],
        "model.gate_cfg.mode": ["8_plus_mod", "8_plus_mod_permute"],
        "model.num_layers": [1, 2],
        "model.pretrained_cfg.should_use": [False],
        "experiment.task.mode": [
            "permute_input_permute_target",
            "rotate_input_permute_target",
        ],
        "model.hidden_layer_cfg.dim": [128],
        "model.should_use_non_linearity": [True, False],
        "model.encoder_cfg.should_share": [False],
        "model.hidden_layer_cfg.should_share": [True],
        "model.decoder_cfg.should_share": [False],
        "model.weight_init.should_do": [True, False],
        "model.weight_init.gain": [1.0],
        "model.weight_init.bias": [0.0],
        "optimizer": ["sgd", "adam"],
        "optimizer.lr": [0.0001],
        # "optimizer.momentum": [0.9],
    }

    return get_overrides_to_test(params_to_test=params_to_test)


@pytest.mark.parametrize(
    "overrides",
    get_overrides_to_test_mnist(),
)
def test_mnist(overrides) -> None:
    check_output_from_cmd(overrides=overrides)


def get_overrides_to_test_cifar10():

    params_to_test = {
        "experiment.num_epochs": [2],
        "experiment.task.num_input_transformations": [10, 80],
        "experiment.task.num_classes_in_selected_dataset": [10],
        "dataloader.train_config.dataloader.batch_size": [
            100,
        ],
        "dataloader": ["filesystem"],
        "experiment": ["k_path_model"],
        "experiment.task.mode": ["rotate_input_permute_target"],
        "model.gate_cfg.mode": ["8_plus_mod", "8_plus_mod_permute"],
        "model.num_layers": [1],
        "model.pretrained_cfg.should_use": [False],
        "model.hidden_layer_cfg.dim": [128],
        "model.should_use_non_linearity": [True, False],
        "model.encoder_cfg.should_share": [False],
        "model.hidden_layer_cfg.should_share": [True],
        "model.decoder_cfg.should_share": [False],
        "model.weight_init.should_do": [True, False],
        "model.weight_init.gain": [1.0],
        "model.weight_init.bias": [0.0],
        "optimizer": ["sgd", "adam"],
        "optimizer.lr": [0.0001],
        # "optimizer.momentum": [0.9],
    }

    return get_overrides_to_test(params_to_test=params_to_test)


@pytest.mark.parametrize(
    "overrides",
    get_overrides_to_test_cifar10(),
)
def test_cifar10(overrides) -> None:
    check_output_from_cmd(overrides=overrides)
