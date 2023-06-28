"""Modules for defining graphs.

These are self-contained graph definitions that hold all the graph-altering
code in graphnet. These modules define what the GNNs sees as input and can be
passed to dataloaders during training and deployment.
"""


from typing import Tuple, Any, List, Optional, Union, Dict, Callable
from abc import abstractmethod, ABC
import torch
from torch_geometric.data import Data
import numpy as np

from graphnet.utilities.config import Configurable
from graphnet.utilities.config import (
    save_model_config,
)  # .graph_config import save_graph_config, GraphConfig
from graphnet.utilities.logging import Logger
from graphnet.models.detector import Detector
from .edges import EdgeDefinition, KNNEdges
from .nodes import NodeDefinition
from graphnet.models import Model


class GraphDefinition(Model):
    """An Abstract class to create graph definitions from."""

    @save_model_config
    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition,
        edge_definition: Optional[EdgeDefinition] = None,
        node_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Construct ´GraphDefinition´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        ´node_definition´ defines the nodes in the graph.

        ´edge_definition´ defines the connectivity of the nodes in the graph.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            node_definition: Definition of nodes.
            edge_definition: Definition of edges. Defaults to None.
            node_feature_names: Names of node feature columns. Defaults to None
            dtype: data type used for node features. e.g. ´torch.float´
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member Variables
        self._detector = detector
        self._edge_definiton = edge_definition
        self._node_definition = node_definition
        if node_feature_names is None:
            # Assume all features in Detector is used.
            node_feature_names = list(self._detector.feature_map().keys())  # type: ignore
        self._node_feature_names = node_feature_names
        if dtype is None:
            dtype = torch.float
        self._dtype = dtype

        # Set Input / Output dimensions
        self._node_definition.set_number_of_inputs(
            node_feature_names=node_feature_names
        )
        self.nb_inputs = len(self._node_feature_names)
        self.nb_outputs = self._node_definition.nb_outputs

    def forward(  # type: ignore
        self,
        node_features: np.array,
        node_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Construct graph as ´Data´ object.

        Args:
            node_features: node features for graph. Shape ´[num_nodes, d]´
            node_feature_names: name of each column. Shape ´[,d]´.
            truth_dicts: Dictionary containing truth labels.
            custom_label_functions: Custom label functions. See https://github.com/graphnet-team/graphnet/blob/main/GETTING_STARTED.md#adding-custom-truth-labels.
            loss_weight_column: Name of column that holds loss weight. Defaults to None.
            loss_weight: Loss weight associated with event. Defaults to None.
            loss_weight_default_value: default value for loss weight. Used in instances where some events have no pre-defined loss weight. Defaults to None.
            data_path: Path to dataset data files. Defaults to None.

        Returns:
            graph
        """
        # Checks
        self._validate_input(
            node_features=node_features, node_feature_names=node_feature_names
        )

        # Transform to pytorch tensor
        node_features = torch.tensor(node_features, dtype=self._dtype)

        # Standardize / Scale  node features
        node_features = self._detector(node_features, node_feature_names)

        # Create graph
        graph = self._node_definition(node_features)

        # Attach number of pulses as static attribute.
        graph.n_pulses = torch.tensor(len(node_features), dtype=torch.int32)

        # Assign edges
        if self._edge_definiton is not None:
            graph = self._edge_definiton(graph)
        else:
            self.warnonce(
                "No EdgeDefinition provided. Graphs will not have edges defined!"
            )

        # Attach data path - useful for Ensemble datasets.
        if data_path is not None:
            graph["dataset_path"] = data_path

        # Attach loss weights if they exist
        graph = self._add_loss_weights(
            graph=graph,
            loss_weight=loss_weight,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
        )

        # Attach default truth labels and node truths
        if truth_dicts is not None:
            graph = self._add_truth(graph=graph, truth_dicts=truth_dicts)

        # Attach custom truth labels
        if custom_label_functions is not None:
            graph = self._add_custom_labels(
                graph=graph, custom_label_functions=custom_label_functions
            )

        # Attach node features as seperate fields. MAY NOT CONTAIN 'x'
        graph = self._add_features_individually(
            graph=graph, node_feature_names=node_feature_names
        )

        return graph

    def _validate_input(
        self, node_features: np.array, node_feature_names: List[str]
    ) -> None:

        # node feature matrix dimension check
        assert node_features.shape[1] == len(node_feature_names)

        # check that provided features for input is the same that the ´Graph´
        # was instantiated with.
        assert len(node_feature_names) == len(
            self._node_feature_names
        ), f"""Input features ({node_feature_names}) is not what {self.__class__.__name__} was instatiated with ({self._node_feature_names})"""
        for idx in range(len(node_feature_names)):
            assert (
                node_feature_names[idx] == self._node_feature_names[idx]
            ), """ Order of node features are not the same."""

    def _add_loss_weights(
        self,
        graph: Data,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
    ) -> Data:
        """Attempt to store a loss weight in the graph for use during training.

        I.e. `graph[loss_weight_column] = loss_weight`

        Args:
            loss_weight: The non-negative weight to be stored.
            graph: Data object representing the event.
            loss_weight_column: The name under which the weight is stored in
                                 the graph.
            loss_weight_default_value: The default value used if
                                        none was retrieved.

        Returns:
            A graph with loss weight added, if available.
        """
        # Add loss weight to graph.
        if loss_weight is not None and loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)
        return graph

    def _add_truth(
        self, graph: Data, truth_dicts: List[Dict[str, Any]]
    ) -> Data:
        """Add truth labels from ´truth_dicts´ to ´graph´.

        I.e. ´graph[key] = truth_dict[key]´


        Args:
            graph: graph where the label will be stored
            truth_dicts: dictionary containing the labels

        Returns:
            graph with labels
        """
        # Write attributes, either target labels, truth info or original
        # features.
        for truth_dict in truth_dicts:
            for key, value in truth_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )
        return graph

    def _add_features_individually(
        self,
        graph: Data,
        node_feature_names: List[str],
    ) -> Data:
        # Additionally add original features as (static) attributes
        graph.features = node_feature_names
        for index, feature in enumerate(node_feature_names):
            if feature not in ["x"]:  # reserved for node features.
                graph[feature] = graph.x[:, index].detach()
            else:
                self.warnonce(
                    """Cannot assign graph['x']. This field is reserved for node features. Please rename your input feature."""
                )
        return graph

    def _add_custom_labels(
        self,
        graph: Data,
        custom_label_functions: Dict[str, Callable[..., Any]],
    ) -> Data:
        # Add custom labels to the graph
        for key, fn in custom_label_functions.items():
            graph[key] = fn(graph)
        return graph
