"""A module containing different graph representations in GraphNeT."""

from typing import List

from .graph_definition import GraphDefinition
from graphnet.models.detector import Detector
from graphnet.models.graphs.edges import KNNEdges
from graphnet.models.graphs.nodes import NodeDefinition


class KNNGraph(GraphDefinition):
    """A Graph representation where Edges are drawn to nearest neighbours."""

    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
    ):
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            nb_nearest_neighbours: Number of edges for each node. Defaults to 8.
            columns: node feature columns used for distance calculation
            . Defaults to [0, 1, 2].
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            node_definition=node_definition,
            edge_definition=KNNEdges(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
        )
