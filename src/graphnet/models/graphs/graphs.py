"""A module containing different graph representations in GraphNeT."""

from typing import List, Optional, Dict, Union, Any
import torch
import numpy as np
from numpy.random import Generator

from torch_geometric.data import Data

from .graph_definition import GraphDefinition
from graphnet.models.detector import Detector
from graphnet.models.graphs.edges import (
    EdgeDefinition,
    KNNEdges,
    KNNDistanceEdges,
)
from graphnet.models.graphs.nodes import NodeDefinition, NodesAsPulses
from graphnet.models.utils import add_full_rrwp, get_rw_landing_probs


class KNNGraph(GraphDefinition):
    """A Graph representation where Edges are drawn to nearest neighbours."""

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
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
            nb_nearest_neighbours: Number of edges for each node.
                Defaults to 8.
            columns: node feature columns used for distance calculation.
                Defaults to [0, 1, 2].
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
        node_definition: Optional[NodeDefinition] = None,
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


class KNNGraphRRWP(GraphDefinition):
    """KNN Graph with relative random walk probabilities (RRWP).

    Identical to KNNGraph, but with five extra fields containing absolute and
    relative positional encoding using RRWP.

    `abs_pe = graph["rrwp"]  # RRWP absolute positional encoding values`
    `rrwp_val = graph["rrwp_val"]  # Non-zero values of the RRWP tensor`
    `rrwp_index = graph["rrwp_index]  # Corresponding row, col indices` `degree
    = graph["deg"]  # Degree of each node (num. of incoming edges)`
    """

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        edge_definition: Optional[EdgeDefinition] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
        walk_length: int = 8,
        **kwargs: Any,
    ) -> None:
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            edge_definition: Definition of edges in the graph.
            input_feature_names: Name of input feature columns.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                Defaults to None.
            nb_nearest_neighbours: Number of edges for each node.
                Defaults to 8.
            columns: node feature columns used for distance calculation.
                Defaults to [0, 1, 2].
            walk_length: number of steps for the random walk.
                Defaults to 8.
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=edge_definition
            or KNNDistanceEdges(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )
        self.walk_length = walk_length

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        **kwargs,
    ) -> Data:
        """Forward pass."""
        graph = super().forward(input_features, input_feature_names, **kwargs)
        graph = add_full_rrwp(graph, walk_length=self.walk_length)

        return graph


class KNNGraphRWSE(GraphDefinition):
    """KNN Graph with random walk structural encoding (RWSE).

    Identical to KNNGraph but with an additional field containing the values
    obtained from RWSE. The encoding can be accessed via

    `rwse = graph["rwse"]  # random walk structural encoding`
    """

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        edge_definition: Optional[EdgeDefinition] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        nb_nearest_neighbours: int = 8,
        columns: List[int] = [0, 1, 2],
        walk_length: int = 8,
        **kwargs: Any,
    ) -> None:
        """Construct k-nn graph representation.

        Args:
            detector: Detector that represents your data.
            node_definition: Definition of nodes in the graph.
            edge_definition: Definition of edges in the graph.
            input_feature_names: Name of input feature columns.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                Defaults to None.
            nb_nearest_neighbours: Number of edges for each node.
                Defaults to 8.
            columns: node feature columns used for distance calculation.
                Defaults to [0, 1, 2].
            walk_length: number of steps for the random walk.
                Defaults to 8.
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=edge_definition
            or KNNEdges(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )
        self.walk_length = walk_length

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        **kwargs,
    ) -> Data:
        """Forward pass."""
        graph = super().forward(input_features, input_feature_names, **kwargs)
        ksteps = torch.arange(1, self.walk_length)
        graph.rwse = get_rw_landing_probs(
            ksteps=ksteps, edge_index=graph.edge_index, edge_weight=None
        )
        return graph


class KNNGraphNoPE(GraphDefinition):
    """KNN Graph with edge distances and no positional encoding."""

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        edge_definition: Optional[EdgeDefinition] = None,
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
            edge_definition: Definition of edges in the graph.
            input_feature_names: Name of input feature columns.
            dtype: data type for node features.
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                Defaults to None.
            nb_nearest_neighbours: Number of edges for each node.
                Defaults to 8.
            columns: node feature columns used for distance calculation.
                Defaults to [0, 1, 2].
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            node_definition=node_definition or NodesAsPulses(),
            edge_definition=edge_definition
            or KNNDistanceEdges(
                nb_nearest_neighbours=nb_nearest_neighbours,
                columns=columns,
            ),
            dtype=dtype,
            input_feature_names=input_feature_names,
            perturbation_dict=perturbation_dict,
            seed=seed,
            **kwargs,
        )

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        **kwargs,
    ) -> Data:
        """Forward pass."""
        graph = super().forward(input_features, input_feature_names, **kwargs)
        return graph
