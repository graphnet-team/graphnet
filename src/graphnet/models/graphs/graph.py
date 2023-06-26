"""Modules for defining graphs.

These are self-contained graph definitions that hold all the graph-altering
code in graphnet. These modules define what the GNNs sees as input and can be
passed to dataloaders during training and deployment.
"""


from typing import Tuple, Any, List, Optional, Union, Dict, Callable
from abc import abstractmethod
import torch
from torch_geometric.data import Data
import numpy as np

from graphnet.utilities.config import save_model_config
from graphnet.models.detector import Detector
from graphnet.models import Model
from graphnet.models.graphs import EdgeDefinition, KNNEdges


class GraphDefinition(Model):
    """An Abstract class to create graph definitions from."""

    @save_model_config
    def __init__(
        self,
        detector: Detector,
        edge_definition: EdgeDefinition,
    ):
        """Construct ´GraphDefinition´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        ´edge_definition´ defines the connectivity of the graph.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            Defaults to None.
            edge_definition: Your choice in edges. Defaults to None.
        """
        # Member Variables
        self._detector = detector
        self._edge_definiton = edge_definition

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def _create_graph(self, x: np.array) -> Data:
        """Problem/model specific graph definition.

        Should not standardize/scale data. May assign edges.

        Args:
            x: node features for a single event

        Returns:
            Data object (a single graph)
        """

    def __call__(
        self,
        node_features: List[Tuple[float, ...]],
        node_feature_names: List[str],
        truth_dicts: List[Dict[str, Any]],
        custom_label_functions: Dict[str, Callable[..., Any]],
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
        # Standardize / Scale  node features
        node_features = self._detector(node_features, node_feature_names)

        # Create graph
        graph = self._create_graph(node_features)

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
        graph = self._add_truth(graph=graph, truth_dicts=truth_dicts)

        # Attach custom truth labels
        graph = self._add_custom_labels(
            graph=graph, custom_label_functions=custom_label_functions
        )

        # Attach node features as seperate fields. MAY NOT CONTAIN 'x'
        graph = self._add_features_individually(
            graph=graph, node_feature_names=node_feature_names
        )

        return graph

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
