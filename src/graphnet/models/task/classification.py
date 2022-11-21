"""Classification-specific `Model` class(es)."""

from torch import Tensor
from torch.nn import Softmax
from graphnet.models.config import save_config

from graphnet.models.task import Task
from typing import Any


class ClassificationTask(Task):
    """Generic Classification task for binary and multi classification."""

    # Requires the same number of features as the number of classes being predicted
    @save_config
    def __init__(self, nb_classes: Any, *args: Any, **kwargs: Any):
        """Initialize of number of class and softmax method."""
        self._nb_classes = nb_classes
        super().__init__(*args, **kwargs)
        self._softmax = Softmax(dim=-1)

    @property
    def nb_inputs(self) -> Tensor:
        """Output from Classification."""
        return self._nb_classes

    def _forward(self, x: Tensor) -> Tensor:
        """Transform latent features into probabilities."""
        return self._softmax(x)
