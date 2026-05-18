"""Concrete :class:`ImageRepresentation` subclasses for common detectors."""

from typing import List, Optional, Any
import torch

from graphnet.models.data_representation.graphs import NodeDefinition
from graphnet.models.detector import Detector, IceCube86, ORCA150

from .image_representation import ImageRepresentation
from .mappings import IC86GridDefinition, ExamplePrometheusGridDefinition


class IC86Image(ImageRepresentation):
    """IceCube-86 images (main array + optional DeepCore tensors)."""

    def __init__(
        self,
        pixel_definition: NodeDefinition,
        input_feature_names: List[str],
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
        string_label: str = "string",
        dom_number_label: str = "dom_number",
        dtype: Optional[torch.dtype] = torch.float,
        detector: Optional[Detector] = None,
        **kwargs: Any,
    ) -> None:
        """Construct `IC86Image`.

        Args:
            pixel_definition: Pulse → DOM row features (:class:`NodeDefinition`).
            input_feature_names: Raw input column names.
            include_lower_dc: Include lower DeepCore grid.
            include_upper_dc: Include upper DeepCore grid.
            string_label: DOM string column in pixel rows.
            dom_number_label: DOM index column in pixel rows.
            dtype: Tensor dtype for images.
            detector: ``IceCube86``; default standardizes all but coordinates.
        """
        if detector is None:
            detector = IceCube86(
                replace_with_identity=input_feature_names,
            )
        else:
            assert isinstance(detector, IceCube86)
        pixel_definition.set_output_feature_names(input_feature_names)
        assert (
            string_label in input_feature_names
        ), f"String label '{string_label}' not in input feature names"
        assert (
            dom_number_label in input_feature_names
        ), f"DOM number label '{dom_number_label}' not in input feature names"

        grid_definition = IC86GridDefinition(
            detector=detector,
            dtype=dtype,
            string_label=string_label,
            dom_number_label=dom_number_label,
            pixel_feature_names=pixel_definition._output_feature_names,
            include_lower_dc=include_lower_dc,
            include_upper_dc=include_upper_dc,
        )

        super().__init__(
            pixel_definition=pixel_definition,
            grid_definition=grid_definition,
            input_feature_names=input_feature_names,
            add_inactive_sensors=False,
            **kwargs,
        )


class ExamplePrometheusImage(ImageRepresentation):
    """Example Prometheus-style single-image layout (tutorial scripts only)."""

    def __init__(
        self,
        pixel_definition: NodeDefinition,
        input_feature_names: List[str],
        string_label: str = "sensor_string_id",
        dom_number_label: str = "sensor_id",
        dtype: Optional[torch.dtype] = torch.float,
        detector: Optional[Detector] = None,
        **kwargs: Any,
    ) -> None:
        """Construct `ExamplePrometheusImage`.

        Args:
            pixel_definition: Pulse → sensor row features (:class:`NodeDefinition`).
            input_feature_names: Raw input column names.
            string_label: String id column in pixel rows.
            dom_number_label: Sensor id column (internal grid key name).
            dtype: Tensor dtype for images.
            detector: ``ORCA150`` by default.
        """
        if detector is None:
            detector = ORCA150(
                replace_with_identity=input_feature_names,
            )

        pixel_definition.set_output_feature_names(input_feature_names)
        assert (
            string_label in input_feature_names
        ), f"String label '{string_label}' not in input feature names"
        assert (
            dom_number_label in input_feature_names
        ), f"DOM number label '{dom_number_label}' not in input feature names"

        grid_definition = ExamplePrometheusGridDefinition(
            detector=detector,
            dtype=dtype,
            string_label=string_label,
            sensor_number_label=dom_number_label,
            pixel_feature_names=pixel_definition._output_feature_names,
        )

        super().__init__(
            pixel_definition=pixel_definition,
            grid_definition=grid_definition,
            input_feature_names=input_feature_names,
            add_inactive_sensors=False,
            **kwargs,
        )
