"""Base GNN-specific `Model` class(es)."""

from abc import abstractmethod
from typing import List

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

    @property
    def coordinate_columns(self) -> List[int]:
        """Return the coordinate column indices.

        `NormalizingFlow` return a tensor of shape
        [n_samples, n_coordinate_columns + jacobian_columns].

        The coordinate column indices are used to slice the tensor `x` from
        the NormalizingFlow.forward() such that x[:,coordinate_columns]
        returns a Tensor containing only the coordinates.
        """
        return self._coordinate_columns

    @property
    def jacobian_columns(self) -> List[int]:
        """Return the coordinate column indices.

        `NormalizingFlow` return a tensor of shape
        [n_samples, n_coordinate_columns + jacobian_columns].

        The jacobian column indices are used to slice the tensor `x` from
        the NormalizingFlow.forward() such that x[:,jacobian_columns]
        returns a Tensor containing only the Jacobian.
        """
        return self._jacobian_columns

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Transform from input distribution into latent distribution."""

    @abstractmethod
    def inverse(self, data: Data) -> Tensor:
        """Transform from latent distribution to input distribution."""
