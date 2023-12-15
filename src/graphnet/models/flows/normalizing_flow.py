"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod

from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import Model


class NormalizingFlow(Model):
    """Base class for all core Normalizing Flow models in GraphNeT."""

    def __init__(self, nb_inputs: int):
        """Construct `NormalizingFlow`."""
        # Base class constructor
        super().__init__()

        # Member variables
        self._nb_inputs = nb_inputs
        self._nb_outputs = nb_inputs  # Normalizing flows are bijective

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
        """Transform from input distribution into latent distribution."""

    @abstractmethod
    def inverse(self, data: Data) -> Tensor:
        """Transform from latent distribution to input distribution."""
