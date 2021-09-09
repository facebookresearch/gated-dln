from src.model import fully_connected
from src.utils.config import DictConfig


class Model(fully_connected.Model):
    """Feedforward Model."""

    def __init__(
        self,
        name: str,
        model_cfg: DictConfig,
        description: str = "",
    ):
        super().__init__(
            name=name,
            model_cfg=model_cfg,
            should_use_non_linearity=True,
            description=description,
        )
