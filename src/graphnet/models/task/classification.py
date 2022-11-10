"""Classification-specific `Model` class(es)."""

from torch import Tensor
from torch.nn import Softmax
from graphnet.models.config import save_config

from graphnet.models.task import Task

class ClassificationTask(Task):
    # Requires the same number of features as the number of classes being predicted
    @save_config
    def __init__(self, nb_inputs: int, *args, **kwargs):
        self._nb_inputs = nb_inputs
        super().__init__(*args, **kwargs)
        self._softmax = Softmax()
    
    @property
    def nb_inputs(self):
        """Number of outputs from Classification."""
        return self._nb_inputs

    def _forward(self, x: Tensor) -> Tensor:
        """Transform latent features into probabilities."""
        return x #self._softmax(x)
