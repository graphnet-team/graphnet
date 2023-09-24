"""A module containing different graph representations in GraphNeT."""

from typing import List, Optional, Dict, Union
import torch
from numpy.random import Generator

from .graph_definition import GraphDefinition
from graphnet.models.detector import Detector
from graphnet.models.graphs.edges import EdgeDefinition, KNNEdges
from graphnet.models.graphs.nodes import NodeDefinition, NodesAsPulses


class KNNGraph(GraphDefinition):
    """A Graph representation where Edges are drawn to nearest neighbours."""

    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition = NodesAsPulses(),
        node_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
    ) -> None:
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            node_feature_names: Name of node features.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
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
            dtype=dtype,
            node_feature_names=node_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
        )
