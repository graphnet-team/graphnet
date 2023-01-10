"""Config classes for the `graphnet.training` module."""

from typing import Any, Dict, List, Union

from graphnet.utilities.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Configuration for all trainings."""

    # Fields
    target: Union[str, List[str]]
    early_stopping_patience: int
    fit: Dict[str, Any]
    dataloader: Dict[str, Any]
