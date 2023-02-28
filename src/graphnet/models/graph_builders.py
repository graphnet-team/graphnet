"""Class(es) for building/connecting graphs."""

from typing import List

import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.data import Data

from graphnet.utilities.config import save_model_config
from graphnet.models.utils import calculate_distance_matrix
from graphnet.models import Model


class GraphBuilder(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""


class KNNGraphBuilder(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph from the k-nearest neighbours."""

    @save_model_config
    def __init__(
        self,
        nb_nearest_neighbours: int,
        columns: List[int] = None,
    ):
        """Construct `KNNGraphBuilder`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if columns is None:
            columns = [0, 1, 2]

        # Member variable(s)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )

        data.edge_index = knn_graph(
            data.x[:, self._columns],
            self._nb_nearest_neighbours,
            data.batch,
        ).to(self.device)

        return data


class RadialGraphBuilder(GraphBuilder):
    """Builds graph from a sphere of chosen radius centred at each node."""

    @save_model_config
    def __init__(
        self,
        radius: float,
        columns: List[int] = None,
    ):
        """Construct `RadialGraphBuilder`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if columns is None:
            columns = [0, 1, 2]

        # Member variable(s)
        self._radius = radius
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )

        data.edge_index = radius_graph(
            data.x[:, self._columns],
            self._radius,
            data.batch,
        ).to(self.device)

        return data


class EuclideanGraphBuilder(
    GraphBuilder
):  # pylint: disable=too-few-public-methods
    """Builds graph according to Euclidean distance between nodes.

    See https://arxiv.org/pdf/1809.06166.pdf.
    """

    @save_model_config
    def __init__(
        self,
        sigma: float,
        threshold: float = 0.0,
        columns: List[int] = None,
    ):
        """Construct `EuclideanGraphBuilder`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if columns is None:
            columns = [0, 1, 2]

        # Member variable(s)
        self._sigma = sigma
        self._threshold = threshold
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )

        xyz_coords = data.x[:, self._columns]

        # Construct block-diagonal matrix indicating whether pulses belong to
        # the same event in the batch
        batch_mask = data.batch.unsqueeze(dim=0) == data.batch.unsqueeze(dim=1)

        distance_matrix = calculate_distance_matrix(xyz_coords)
        affinity_matrix = torch.exp(
            -0.5 * distance_matrix**2 / self._sigma**2
        )

        # Use softmax to normalise all adjacencies to one for each node
        exp_row_sums = torch.exp(affinity_matrix).sum(axis=1)
        weighted_adj_matrix = torch.exp(
            affinity_matrix
        ) / exp_row_sums.unsqueeze(dim=1)

        # Only include edges with weights that exceed the chosen threshold (and
        # are part of the same event)
        sources, targets = torch.where(
            (weighted_adj_matrix > self._threshold) & (batch_mask)
        )
        edge_weights = weighted_adj_matrix[sources, targets]

        data.edge_index = torch.stack((sources, targets))
        data.edge_weight = edge_weights

        return data


# class MinkowskiGraphBuilder(GraphBuilder):
#    ...
