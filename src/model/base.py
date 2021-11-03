import pathlib

import numpy as np
from torch import nn
from xplogger.logbook import LogBook

from src.checkpoint import utils as checkpoint_utils
from src.utils.config import DictConfig


class Model(nn.Module):
    """Basic component (for building the agent) that every other component should extend.
    It inherits `torch.nn.Module`.
    """

    def __init__(
        self,
        name: str,
        model_cfg: DictConfig,
        description: str = "",
    ):
        super().__init__()
        self.name = name
        if description:
            self.description = description
        else:
            self.description = (
                "This is the base class for all the components/models. "
                "All the other components/models should extend this class. "
                "It is not to be used directly."
            )
        self.model_cfg = model_cfg

    def get_param_count(self) -> int:
        """Count the number of params"""
        return sum((np.prod(p.size()) for p in self.parameters()))

    def get_trainable_param_count(self) -> int:
        """Count the number of trainable params"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum((np.prod(p.size()) for p in model_parameters))

    def freeze_weights(self) -> None:
        """Freeze the model"""
        for param in self.parameters():
            param.requires_grad = False

    def save(
        self,
        name: str,
        save_dir_path: pathlib.Path,
        step: int,
        retain_last_n: int,
        logbook: LogBook,
    ) -> None:
        return checkpoint_utils.save_model(
            model=self,
            name=name,
            save_dir_path=save_dir_path,
            step=step,
            retain_last_n=retain_last_n,
            logbook=logbook,
        )

    def load(self, name: str, save_dir: str, step: int, logbook: LogBook) -> nn.Module:
        return checkpoint_utils.load_model(  # type: ignore[return-value]
            model=self, name=name, save_dir=save_dir, step=step, logbook=logbook
        )
        # mpyp error: Incompatible return value type (got Module, expected "Model")  [return-value]
