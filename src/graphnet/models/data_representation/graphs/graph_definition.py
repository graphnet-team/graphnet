"""Modules for defining graphs.

These are self-contained graph definitions that hold all the graph-
altering code in graphnet. These modules define what graph-based models
sees as input and can be passed to dataloaders during training and
deployment.
"""

from typing import List, Optional, Dict, Union, Any, Callable
import torch
from numpy.random import Generator
import numpy as np

from graphnet.models.detector import Detector
from .edges import EdgeDefinition
from .nodes import NodeDefinition, NodesAsPulses
from graphnet.models.data_representation import DataRepresentation
from torch_geometric.data import Data


class GraphDefinition(DataRepresentation):
    """An Abstract class to create graph definitions from."""

    def __init__(
        self,
        detector: Detector,
        node_definition: Optional[NodeDefinition] = None,
        edge_definition: Optional[EdgeDefinition] = None,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
        sort_by: Optional[str] = None,
        repeat_labels: bool = False,
        add_static_features: bool = True,
    ):
        """Construct ´GraphDefinition´. The ´detector´ holds.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        ´node_definition´ defines the nodes in the graph.

        ´edge_definition´ defines the connectivity of the nodes in the graph.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            node_definition: Definition of nodes. Defaults to NodesAsPulses.
            edge_definition: Definition of edges. Defaults to None.
            input_feature_names: Names of each column in expected input data
                that will be built into a graph. If not provided,
                it is automatically assumed that all features in `Detector` is
                used.
            dtype: data type used for node features. e.g. ´torch.float´
            perturbation_dict: Dictionary mapping a feature name to a standard
                               deviation according to which the values for this
                               feature should be randomly perturbed. Defaults
                               to None.
            seed: seed or Generator used to randomly sample perturbations.
                  Defaults to None.
            add_inactive_sensors: If True, inactive sensors will be appended
                to the graph with padded pulse information. Defaults to False.
            sensor_mask: A list of sensor id's to be masked from the graph. Any
                sensor listed here will be removed from the graph.
                    Defaults to None.
            string_mask: A list of string id's to be masked from the graph.
                Defaults to None.
            sort_by: Name of node feature to sort by. Defaults to None.
            repeat_labels: If True, labels will be repeated to match the
                the number of rows in the output of the GraphDefinition.
                Defaults to False.
            add_static_features: If True, the original features will be
                added as static attributes to the graph. Defaults to True.
        """
        # Base class constructor
        super().__init__(
            detector=detector,
            input_feature_names=input_feature_names,
            dtype=dtype,
            perturbation_dict=perturbation_dict,
            seed=seed,
            add_inactive_sensors=add_inactive_sensors,
            sensor_mask=sensor_mask,
            string_mask=string_mask,
            repeat_labels=repeat_labels,
        )

        if node_definition is None:
            node_definition = NodesAsPulses()
        self._node_definition = node_definition
        self._edge_definition = edge_definition
        if self._edge_definition is None:
            self.warning_once(
                """No EdgeDefinition given. Graphs will not have edges!"""
            )

        # Sorting
        if sort_by is not None:
            assert isinstance(sort_by, str)
            try:
                sort_by = self.output_feature_names.index(  # type: ignore
                    sort_by
                )  # type: ignore
            except ValueError as e:
                self.error(
                    f"{sort_by} not in node "
                    f"features {self.output_feature_names}."
                )
                raise e
        self._sort_by = sort_by
        self._add_static_features = add_static_features

        # make sure output feature names are set also in node definition
        self._set_output_feature_names(self._input_feature_names)

    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Set the final data output feature names."""
        # Set input data column names for node definition
        self._node_definition.set_output_feature_names(input_feature_names)
        return self._node_definition._output_feature_names

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        """Construct graph as ´Data´ object.

        Args:
            input_features: Input features for graph construction.
                Shape ´[num_rows, d]´
            input_feature_names: name of each column. Shape ´[,d]´.
            truth_dicts: Dictionary containing truth labels.
            custom_label_functions: Custom label functions.
            loss_weight_column: Name of column that holds loss weight.
                                Defaults to None.
            loss_weight: Loss weight associated with event. Defaults to None.
            loss_weight_default_value: default value for loss weight.
                    Used in instances where some events have
                    no pre-defined loss weight. Defaults to None.
            data_path: Path to dataset data files. Defaults to None.

        Returns:
            graph
        """
        data = super().forward(
            input_features=input_features,
            input_feature_names=input_feature_names,
            truth_dicts=truth_dicts,
            custom_label_functions=custom_label_functions,
            loss_weight_column=loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=loss_weight_default_value,
            data_path=data_path,
        )
        # Create graph & get new node feature names
        data.x = self._node_definition(data.x)
        if self._sort_by is not None:
            data.x = data.x[data.x[:, self._sort_by].sort()[1]]

        # Enforce dtype
        data.x = data.x.type(self.dtype)

        # Assign edges
        if self._edge_definition is not None:
            data = self._edge_definition(data)

        if self._add_static_features:
            data = self._add_features_individually(
                data,
                self.output_feature_names,
            )
        return data

    def _add_features_individually(
        self,
        data: Data,
        data_feature_names: List[str],
    ) -> Data:
        # Additionally add original features as (static) attributes
        data.features = data_feature_names
        for index, feature in enumerate(data_feature_names):
            if feature not in ["x"]:  # reserved for graph features.
                data[feature] = data.x[:, index].detach()
            else:
                self.warning_once(
                    """Cannot assign data['x']. This field is reserved for
                      graph node features. Please rename your input feature."""
                )  # noqa

        return data

    def _label_repeater(self, label: torch.Tensor, data: Data) -> torch.Tensor:
        """Handle the label repetition.

        Not a classmethod because it is not necessary for all data
        representations.
        """
        return label.repeat(data.x.shape[0], 1)
