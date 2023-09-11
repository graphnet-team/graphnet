"""Class(es) for building/connecting graphs."""

from typing import List
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.utilities.config import save_model_config
from graphnet.models import Model


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    @save_model_config
    def __init__(self) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @final
    def forward(self, x: torch.tensor) -> Data:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.

        Returns:
            graph: a graph without edges
        """
        graph = self._construct_nodes(x)
        return graph

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return self.nb_inputs

    @final
    def set_number_of_inputs(self, node_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            node_feature_names: name of each node feature column.
        """
        assert isinstance(node_feature_names, list)
        self.nb_inputs = len(node_feature_names)

    @abstractmethod
    def _construct_nodes(self, x: torch.tensor) -> Data:
        """Construct nodes from raw node features ´x´.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.

        Returns:
            graph: graph without edges.
        """


class NodesAsPulses(NodeDefinition):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        return Data(x=x)
