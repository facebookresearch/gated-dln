from src.model.moe import sparsely_connected
from src.utils.config import DictConfig


class Model(sparsely_connected.Model):
    """Feedforward Model."""

    def __init__(
        self,
        name: str,
        model_cfg: DictConfig,
        loss_fn_cfg: DictConfig,
        description: str = "",
    ):
        super().__init__(
            name=name,
            model_cfg=model_cfg,
            loss_fn_cfg=loss_fn_cfg,
            should_use_non_linearity=True,
            description=description,
        )
