"""A module containing different image representations in GraphNeT."""

from typing import List, Optional, Any
import torch

from graphnet.models.data_representation.graphs import NodeDefinition
from graphnet.models.detector import IceCube86

from .image_definition import ImageDefinition
from .mappings import IC86DNNMapping


class IC86DNNImage(ImageDefinition):
    """Class creating a image for IC86 DNN data."""

    def __init__(
        self,
        node_definition: NodeDefinition,
        input_feature_names: List[str],
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
        dtype: Optional[torch.dtype] = torch.float,
        **kwargs: Any,
    ) -> None:
        """Construct `IC86DNNImage`.

        Args:
            node_definition: Definition of nodes.
            input_feature_names: Names of each column in expected input data
                that will be built into a image.
            include_lower_dc: If True, the lower DeepCore will be included.
            include_upper_dc: If True, the upper DeepCore will be included.
            dtype: data type used for node features. e.g. ´torch.float´
        """
        node_definition.set_output_feature_names(input_feature_names)
        dom_labels = node_definition._cluster_on

        # Base class constructor
        pixel_mapping = IC86DNNMapping(
            dom_pos_names=dom_labels,
            pixel_feature_names=node_definition._output_feature_names,
            include_lower_dc=include_lower_dc,
            include_upper_dc=include_upper_dc,
            dtype=dtype,
        )
        super().__init__(
            detector=IceCube86(replace_with_identity=dom_labels),
            node_definition=node_definition,
            pixel_mapping=pixel_mapping,  # PixelMapping,
            input_feature_names=input_feature_names,
            add_inactive_sensors=False,
            **kwargs,
        )
