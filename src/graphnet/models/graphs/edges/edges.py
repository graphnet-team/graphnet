"""Class(es) for building/connecting graphs."""

from typing import List
from abc import abstractmethod

import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.data import Data

from graphnet.models.utils import calculate_distance_matrix
from graphnet.models import Model


class EdgeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def forward(self, graph: Data) -> Data:
        """Construct edges based on problem specific implementation of.

        ´_construct_edges´

        Args:
            graph: a graph without edges

        Returns:
            graph: a graph with edges
        """
        if graph.edge_index is not None:
            self.warning_once(
                "GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )
        return self._construct_edges(graph)

    @abstractmethod
    def _construct_edges(self, graph: Data) -> Data:
        """Construct edges and assign them to the graph.

        I.e. ´graph.edge_index = edge_index´.

        Args:
            graph: graph without edges

        Returns:
            graph: graph with edges assigned.
        """


class KNNEdges(EdgeDefinition):  # pylint: disable=too-few-public-methods
    """Builds edges from the k-nearest neighbours."""

    def __init__(
        self,
        nb_nearest_neighbours: int,
        columns: List[int] = [0, 1, 2],
    ):
        """K-NN Edge definition.

        Will connect nodes together with their ´nb_nearest_neighbours´
        nearest neighbours in the feature space given by ´columns´.

        Args:
            nb_nearest_neighbours: number of neighbours.
            columns: Node features to use for distance calculation.
            Defaults to [0,1,2].
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member variable(s)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns

    def _construct_edges(self, graph: Data) -> Data:
        """Define K-NN edges."""
        graph.edge_index = knn_graph(
            graph.x[:, self._columns],
            self._nb_nearest_neighbours,
            graph.batch,
        ).to(self.device)

        return graph


class RadialEdges(EdgeDefinition):
    """Builds graph from a sphere of chosen radius centred at each node."""

    def __init__(
        self,
        radius: float,
        columns: List[int] = [0, 1, 2],
    ):
        """Radial edges.

        Connects each node to other nodes that are within a sphere of
        radius ´r´ centered at the node. The feature space of ´r´ is defined
        by ´columns´

        Args:
            radius: radius of sphere
            columns: columns of the node feature matrix used.
            Defaults to [0,1,2].
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member variable(s)
        self._radius = radius
        self._columns = columns

    def _construct_edges(self, graph: Data) -> Data:
        """Define radial edges."""
        graph.edge_index = radius_graph(
            graph.x[:, self._columns],
            self._radius,
            graph.batch,
        ).to(self.device)

        return graph


class EuclideanEdges(EdgeDefinition):  # pylint: disable=too-few-public-methods
    """Builds edges according to Euclidean distance between nodes.

    See https://arxiv.org/pdf/1809.06166.pdf.
    """

    def __init__(
        self,
        sigma: float,
        threshold: float = 0.0,
        columns: List[int] = [0, 1, 2],
    ):
        """Construct `EuclideanEdges`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member variable(s)
        self._sigma = sigma
        self._threshold = threshold
        self._columns = columns

    def _construct_edges(self, graph: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        if graph.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )

        xyz_coords = graph.x[:, self._columns]

        # Construct block-diagonal matrix indicating whether pulses belong to
        # the same event in the batch
        batch_mask = graph.batch.unsqueeze(dim=0) == graph.batch.unsqueeze(
            dim=1
        )

        distance_matrix = calculate_distance_matrix(xyz_coords)
        affinity_matrix = torch.exp(-0.5 * distance_matrix**2 / self._sigma**2)

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

        graph.edge_index = torch.stack((sources, targets))
        graph.edge_weight = edge_weights

        return graph
