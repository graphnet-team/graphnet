"""Modules for testing Images and Mappings."""

from typing import List, Optional, Any
import torch
from .mappings import IC86DNNMapping
from .image_definition import ImageDefinition
from graphnet.models.detector import IceCube86
from graphnet.models.data_representation.graphs import NodeDefinition
from torch_geometric.data import Data


class TestImageIC86Mapping(ImageDefinition):
    """Class creating a test image for IC86 DNN data."""

    def __init__(
        self,
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
        input_feature_names: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
            "string",
            "dom_number",
        ],
        dtype: Optional[torch.dtype] = torch.float,
        **kwargs: Any,
    ) -> None:
        """Construct `TestImageIC86Mapping`.

        Args:
            include_lower_dc: If True, the lower DeepCore will be included.
            include_upper_dc: If True, the upper DeepCore will be included.
            input_feature_names: Names of each column in expected input data
                that will be built into a image.
            dtype: data type used for node features. e.g. ´torch.float´
        """
        node_definition = TestPixel()
        node_definition.set_output_feature_names(input_feature_names)
        dom_labels = ["dom_x", "dom_y", "dom_z"]

        # Base class constructor
        pixel_mapping = IC86DNNMapping(
            string_label="string",
            dom_number_label="dom_number",
            pixel_feature_names=node_definition._output_feature_names,
            include_lower_dc=include_lower_dc,
            include_upper_dc=include_upper_dc,
            dtype=dtype,
        )
        super().__init__(
            detector=IceCube86(
                replace_with_identity=dom_labels + ["string", "dom_number"]
            ),
            node_definition=node_definition,
            pixel_mapping=pixel_mapping,  # PixelMapping,
            input_feature_names=input_feature_names,
            add_inactive_sensors=False,
            **kwargs,
        )


class TestPixel(NodeDefinition):
    """Represent pixels as clusters with percentile summary pixel features.

    If `cluster_on` is set to the xyz coordinates of DOMs
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each pixel will be a
    unique DOM and the pulse information (charge, time) is summarized using
    percentiles.
    """

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        assert set(input_feature_names) == set(
            ["dom_x", "dom_y", "dom_z", "string", "dom_number"]
        )
        return input_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        # Cast to Numpy
        return x
