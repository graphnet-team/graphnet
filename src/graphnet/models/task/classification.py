"""Classification-specific `Model` class(es)."""

from typing import Any

import torch
from torch import Tensor

from graphnet.models.task import Task, IdentityTask


class MulticlassClassificationTask(IdentityTask):
    """General task for classifying any number of classes.

    Requires the same number of input features as the number of classes being
    predicted. Returns the untransformed latent features, which are interpreted
    as the logits for each class being classified.
    """


class BinaryClassificationTask(Task):
    """Performs binary classification."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1
    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]

    def _forward(self, x: Tensor) -> Tensor:
        # transform probability of being muon
        return torch.sigmoid(x)


class BinaryClassificationTaskLogits(Task):
    """Performs binary classification form logits."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1
    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]

    def _forward(self, x: Tensor) -> Tensor:
        return x
