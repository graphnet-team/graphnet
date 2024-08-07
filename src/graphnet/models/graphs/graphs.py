"""A module containing different graph representations in GraphNeT."""

from typing import List, Optional, Dict, Union, Any
import torch
from numpy.random import Generator

from .graph_definition import GraphDefinition
from graphnet.models.detector import Detector
from graphnet.models.graphs.edges import KNNEdges
from graphnet.models.graphs.nodes import NodeDefinition, NodesAsPulses


class KNNGraph(GraphDefinition):
    """A Graph representation where Edges are drawn to nearest neighbours."""

    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
        **kwargs: Any,
    ) -> None:
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            input_feature_names: Name of input feature columns.
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
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=KNNEdges(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )


class EdgelessGraph(GraphDefinition):
    """A Data representation without edge assignment.

    I.e the resulting representation is created without an EdgeDefinition.
    """

    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        **kwargs: Any,
    ) -> None:
        """Construct isolated nodes graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            input_feature_names: Name of input feature columns.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=None,
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )
