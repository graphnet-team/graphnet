"""A module containing different image representations in GraphNeT."""

from typing import List, Optional, Any
import torch

from graphnet.models.data_representation.graphs import NodeDefinition
from graphnet.models.detector import Detector, IceCube86

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
        string_label: str = "string",
        dom_number_label: str = "dom_number",
        dtype: Optional[torch.dtype] = torch.float,
        detector: Optional[Detector] = None,
        **kwargs: Any,
    ) -> None:
        """Construct `IC86DNNImage`.

        Args:
            node_definition: Definition of nodes.
            input_feature_names: Names of each column in expected input data
                that will be built into a image.
            include_lower_dc: If True, the lower DeepCore will be included.
            include_upper_dc: If True, the upper DeepCore will be included.
            string_label: The label for the string number in the data.
            dom_number_label: The label for the DOM number in the data.
            dtype: data type used for node features. e.g. ´torch.float´
            detector: The corresponding ´Detector´ representing the data.
        """
        # Default detector with unstandardized input features
        if detector is None:
            detector = IceCube86(
                replace_with_identity=input_feature_names,
            )
        else:
            assert isinstance(detector, IceCube86)
        node_definition.set_output_feature_names(input_feature_names)
        assert (
            string_label in input_feature_names
        ), f"String label '{string_label}' not in input feature names"
        assert (
            dom_number_label in input_feature_names
        ), f"DOM number label '{dom_number_label}' not in input feature names"

        # Base class constructor
        pixel_mapping = IC86DNNMapping(
            string_label=string_label,
            dom_number_label=dom_number_label,
            pixel_feature_names=node_definition._output_feature_names,
            include_lower_dc=include_lower_dc,
            include_upper_dc=include_upper_dc,
            dtype=dtype,
        )

        super().__init__(
            detector=detector,
            node_definition=node_definition,
            pixel_mapping=pixel_mapping,  # PixelMapping,
            input_feature_names=input_feature_names,
            add_inactive_sensors=False,
            **kwargs,
        )
