"""Base CNN-specific `Model` class(es)."""

from abc import abstractmethod

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import Model


class CNN(Model):
    """Base class for all core CNN models in graphnet."""

    def __init__(self, nb_inputs: int, nb_outputs: int) -> None:
        """Construct `CNN`."""
        # Base class constructor
        super().__init__()

        # Member variables
        self._nb_inputs = nb_inputs
        self._nb_outputs = nb_outputs

    @property
    def nb_inputs(self) -> int:
        """Return number of input features."""
        return self._nb_inputs

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return self._nb_outputs

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass in model."""
