"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import Model
from graphnet.utilities.config import save_model_config


class GNN(Model):
    """Base class for all core GNN models in graphnet."""

    @save_model_config
    def __init__(self, nb_inputs: int, nb_outputs: int) -> None:
        """Construct `GNN`."""
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
