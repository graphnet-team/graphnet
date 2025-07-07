"""Modules for defining images.

These are self-contained image definitions that hold all the image-
altering code in graphnet. These modules define what image-based models
sees as input and can be passed to dataloaders during training and
deployment.
"""

from typing import List, Optional, Dict, Union, Any, Callable
import torch
import numpy as np
from numpy.random import Generator

from graphnet.models.detector import Detector
from graphnet.models.data_representation import DataRepresentation
from graphnet.models.data_representation.graphs import NodeDefinition
from torch_geometric.data import Data
from .mappings import PixelMapping


class ImageDefinition(DataRepresentation):
    """An Abstract class to create Imagedefinitions from."""

    def __init__(
        self,
        detector: Detector,
        node_definition: NodeDefinition,
        pixel_mapping: PixelMapping,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
    ):
        """Construct `ImageDefinition`.

        ´Detector´-specific code. E.g. scaling/standardization and geometry
        tables.

        ´node_definition´ defines the processing of raw data.

        ´pixel_mapping´ defines the mapping of the processed data to images.

        NOTE: some pixel_mappings require specific node_definitions.

        Args:
            detector: The corresponding ´Detector´ representing the data.
            node_definition: Definition of nodes.
            pixel_mapping: Definition of Mapping form nodes to pixels.
            input_feature_names: Names of each column in expected input data
                that will be built into a image. If not provided,
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
            repeat_labels=False,
        )

        self._node_definition = node_definition
        self._pixel_mapping = pixel_mapping

    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Set the final data output feature names."""
        # Set input data column names for pixel definition
        self._node_definition.set_output_feature_names(input_feature_names)

        # get output data column names for pixel mapping
        self._pixel_mapping._set_image_feature_names(
            self._node_definition._output_feature_names
        )
        return self._pixel_mapping.image_feature_names

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

        # data processing
        data.x = self._node_definition(data.x)

        # set data type
        data.x = data.x.type(self.dtype)

        # create image
        data = self._pixel_mapping(data, self.output_feature_names)

        if not isinstance(data.x, list):
            data.x = [data.x]

        nb_nodes = []
        for i, x in enumerate(data.x):
            data.x[i] = x.type(self.dtype)

            # setting number of nodes as product of C*(D*)H*W
            nb_nodes.append(np.prod(list(data.x[i].size()[2:])))

        # set num_nodes equals number of pixels in all imagessurpress warning
        data.num_nodes = torch.tensor(np.sum(nb_nodes))

        return data
