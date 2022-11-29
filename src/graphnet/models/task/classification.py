"""Classification-specific `Model` class(es)."""

from torch import Tensor, sigmoid
from torch.nn import Softmax
from graphnet.utilities.config.model_config import save_model_config

from graphnet.models.task import Task
from typing import Any


class ClassificationTask(Task):
    """Generic Classification task for binary and multi classification."""

    # Requires the same number of features as the number of classes being predicted
    @save_model_config
    def __init__(self, nb_classes: int, *args: Any, **kwargs: Any):
        """Initialize of number of class and softmax method."""
        self._nb_classes = nb_classes
        super().__init__(*args, **kwargs)
        # TODO: find a way to remove the transform_inference argument and use softmax here.
        self._softmax = None

    @property
    def nb_inputs(self) -> Tensor:
        """Output from Classification."""
        return self._nb_classes

    def _forward(self, x: Tensor) -> Tensor:
        """Transform latent features into probabilities."""
        return x


class BinaryClassificationTask(Task):
    """Performs binary classification."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # transform probability of being muon
        return sigmoid(x)


class BinaryClassificationTaskLogits(Task):
    """Performs binary classification form logits."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return x
