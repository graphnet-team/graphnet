"""Image CNN input pipeline: pulses → pixels → grids / tensors."""

from typing import List, Optional, Dict, Tuple, Union, Any, Callable
import torch
import numpy as np
from numpy.random import Generator

from graphnet.models.data_representation import DataRepresentation
from graphnet.models.data_representation.graphs import NodeDefinition
from torch_geometric.data import Data
from .mappings import GridDefinition


class ImageRepresentation(DataRepresentation):
    """Compose a pixel definition with a detector grid for CNN inputs.

    A :class:`~graphnet.models.data_representation.graphs.nodes.NodeDefinition`
    acts as **pixel definition**: pulses ``X`` are aggregated into unordered
    pixel rows ``P`` (the same abstraction as graph nodes, without requiring
    graph terminology for CNN users).

    A :class:`GridDefinition` defines detector-bound orthonormal grid shape(s)
    and lookup table(s); its :meth:`~GridDefinition.forward` scatters ``P``
    into image tensor(s).

    The :class:`~graphnet.models.detector.detector.Detector` is taken from
    ``grid_definition.detector`` so the grid matches the preprocessing geometry.
    """

    def __init__(
        self,
        pixel_definition: NodeDefinition,
        grid_definition: GridDefinition,
        input_feature_names: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = torch.float,
        perturbation_dict: Optional[Dict[str, float]] = None,
        seed: Optional[Union[int, Generator]] = None,
        add_inactive_sensors: bool = False,
        sensor_mask: Optional[List[int]] = None,
        string_mask: Optional[List[int]] = None,
    ):
        """Construct `ImageRepresentation`.

        Args:
            pixel_definition: Pulse-level features → one row per pixel/DOM.
            grid_definition: Pixel keys + voxel indices + scatter into images.
            input_feature_names: Column names in raw pulse tables. If omitted,
                the detector's feature list is used.
            dtype: Feature dtype (e.g. ``torch.float``).
            perturbation_dict: Optional feature noise (see ``DataRepresentation``).
            seed: RNG for perturbations.
            add_inactive_sensors: Pad inactive sensors when True.
            sensor_mask: Drop these sensor IDs.
            string_mask: Drop these string IDs.

        Note:
            ``pixel_definition`` output columns must match what
            ``grid_definition`` expects (including key fields in
            :attr:`GridDefinition.map_pixels_by`).
        """
        super().__init__(
            detector=grid_definition.detector,
            input_feature_names=input_feature_names,
            dtype=dtype,
            perturbation_dict=perturbation_dict,
            seed=seed,
            add_inactive_sensors=add_inactive_sensors,
            sensor_mask=sensor_mask,
            string_mask=string_mask,
            repeat_labels=False,
        )
        self._pixel_definition = pixel_definition
        self._grid_definition = grid_definition

    @property
    def shape(self) -> List[List[int]]:
        """Channel-spatial layout per image tensor (see ``GridDefinition``)."""
        return self._grid_definition.shape

    def single_image_spatial_shape(self) -> Tuple[int, int, int]:
        """Return spatial size as ``(height, width, depth)`` for one 3D image.

        Raises:
            ValueError: If ``shape`` does not describe exactly one four-axis
                layout (channels plus three spatial axes).
        """
        layouts = self.shape
        if len(layouts) != 1:
            raise ValueError(
                "Expected a single-image data representation (one shape "
                f"entry), got {len(layouts)}. For multi-image inputs, build "
                "the backbone explicitly for each tensor."
            )
        layout = layouts[0]
        if len(layout) != 4:
            raise ValueError(
                "Expected each image layout as "
                "[num_channels, height, width, depth]; "
                f"got {layout!r}."
            )
        return (layout[1], layout[2], layout[3])

    def _set_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Sync pixel columns and grid output names."""
        self._pixel_definition.set_output_feature_names(input_feature_names)
        self._grid_definition._set_image_feature_names(
            self._pixel_definition._output_feature_names
        )
        return self._grid_definition.image_feature_names

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
        """Build a ``Data`` object with image tensor(s) on ``x``."""
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
        data.x = self._pixel_definition(data.x)
        data.x = data.x.type(self.dtype)
        data = self._grid_definition(data, self.output_feature_names)
        if not isinstance(data.x, list):
            data.x = [data.x]
        nb_nodes = []
        for i, x in enumerate(data.x):
            data.x[i] = x.type(self.dtype)
            nb_nodes.append(np.prod(list(data.x[i].size()[2:])))
        data.num_nodes = torch.tensor(np.sum(nb_nodes))
        return data
