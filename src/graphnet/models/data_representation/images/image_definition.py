"""Modules for defining images.

These are self-contained image definitions that hold all the image-altering
code in graphnet. These modules define what image-based models sees as input
and can be passed to dataloaders during training and deployment.
"""

from typing import List, Optional, Dict, Union, Tuple
import torch
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

    def _create_data(
        self, input_features: torch.Tensor
    ) -> Tuple[Data, List[str]]:
        # Create image & get new pixel feature names
        data, data_feature_names = self._node_definition(input_features)

        data.x = data.x.type(self.dtype)

        data = self._pixel_mapping(data, data_feature_names)

        if not isinstance(data.x, list):
            data.x = [data.x]

        for i, x in enumerate(data.x):
            data.x[i] = x.type(self.dtype)

        return data, data_feature_names
