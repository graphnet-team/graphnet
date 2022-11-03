"""Classification-specific `Model` class(es)."""

import torch
from torch import Tensor

from graphnet.models.task import Task


class BinaryClassificationTask(Task):
    """Performs binary classification."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # transform probability of being muon
        return torch.sigmoid(x)


class BinaryClassificationTaskLogits(Task):
    """Performs binary classification form logits."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        return x
