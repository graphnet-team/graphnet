"""Detector-specific grid layouts: lookup tables, shapes, and scatter into tensors.

Each :class:`GridDefinition` is bound to a :class:`~graphnet.models.detector.Detector`.
:class:`~graphnet.models.data_representation.images.image_representation.ImageRepresentation`
calls :meth:`GridDefinition.forward` to place pixel rows into image tensor(s).
"""

from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphnet.models import Model
from graphnet.models.detector import Detector

from .cnn_mapping_tables import (
    MAT_AX0_COL,
    MAT_AX1_COL,
    MAT_AX2_COL,
    PROMETHEUS_CNN_MAPPING_ROWS,
    build_ic86_cnn_mapping,
)


class GridDefinition(Model):
    """Detector-specific orthonormal image grid(s).

    Holds tensor shapes and tables that map pixel keys to voxel indices.
    """

    def __init__(
        self,
        detector: Detector,
        pixel_feature_names: List[str],
    ) -> None:
        """Construct `GridDefinition`.

        Args:
            detector: Geometry this grid is defined for (CNN grids are
                detector-specific).
            pixel_feature_names: Column names expected on each pixel row,
                including keys listed in :attr:`map_pixels_by`.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._detector = detector
        self._set_image_feature_names(pixel_feature_names)

    @property
    def detector(self) -> Detector:
        """Detector instance this grid belongs to."""
        return self._detector

    @property
    @abstractmethod
    def map_pixels_by(self) -> List[str]:
        """Feature columns that join pixel rows to :meth:`mappings`."""

    @abstractmethod
    def mappings(self) -> List[pd.DataFrame]:
        """DataFrame(s) keyed by ``map_pixels_by`` with voxel index columns."""

    @abstractmethod
    def forward(self, data: Data, data_feature_names: List[str]) -> Data:
        """Scatter pixel features into shaped tensors (see subclasses)."""

    @abstractmethod
    def _set_image_feature_names(self, input_feature_names: List[str]) -> None:
        """Set the final image feature names."""
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(
        self,
    ) -> List[List[int]]:
        """Return the shape of the output images as a list of tuples.

        In the dimensions (F,D,H,W) where F is the number of features
        per pixel. And D,H,W are the dimension of the image
        """
        pass


class IC86GridDefinition(GridDefinition):
    """IceCube-86 CNN grid (main array + DeepCore) layouts and lookup table."""

    def __init__(
        self,
        detector: Detector,
        dtype: torch.dtype,
        pixel_feature_names: List[str],
        string_label: str = "string",
        dom_number_label: str = "dom_number",
        include_main_array: bool = True,
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
    ) -> None:
        """Construct `IC86GridDefinition`.

        The mapping from (string, dom_number) to a position in the
        resulting images is generated programmatically at instantiation
        time from the IceCube86 detector geometry, so no auxiliary file
        is required.

        Args:
            detector: ``IceCube86`` instance (grid is fixed to that geometry).
            dtype: data type used for node features. e.g. ´torch.float´
            string_label: Name of the feature corresponding
                to the DOM string number. Values Integers between 1 - 86
            dom_number_label: Name of the feature corresponding
                to the DOM number (1 - 60). Values Integers between 1 - 60
                where 1 is the dom with the highest z coordinate.
            pixel_feature_names: Names of each column in expected input data
                that will be built into a image.
            include_main_array: If True, the main array will be included.
            include_lower_dc: If True, the lower DeepCore will be included.
            include_upper_dc: If True, the upper DeepCore will be included.

        Raises:
            ValueError: If no array type is included.

        NOTE: Expects input data to be DOMs with aggregated features.
        """
        if not np.any(
            [include_main_array, include_lower_dc, include_upper_dc]
        ):
            raise ValueError("Include at least one array type.")

        self._dtype = dtype
        self._string_label = string_label
        self._dom_number_label = dom_number_label
        self._pixel_feature_names = pixel_feature_names

        self._set_indices(pixel_feature_names, dom_number_label, string_label)

        self._nb_cnn_features = (
            len(pixel_feature_names) - 2
        )  # 2 for string and dom_number

        self._include_main_array = include_main_array
        self._include_lower_dc = include_lower_dc
        self._include_upper_dc = include_upper_dc

        self._mapping = build_ic86_cnn_mapping(
            string_label=string_label,
            dom_number_label=dom_number_label,
        )
        super().__init__(
            detector=detector, pixel_feature_names=pixel_feature_names
        )

    @property
    def map_pixels_by(self) -> List[str]:
        """String and DOM identifiers used for voxel lookup."""
        return [self._string_label, self._dom_number_label]

    def mappings(self) -> List[pd.DataFrame]:
        """Return the single combined lookup table for all sub-images."""
        return [self._mapping]

    def _set_indices(
        self,
        feature_names: List[str],
        dom_number_label: str,
        string_label: str,
    ) -> None:
        """Set the indices for the features."""
        self._cnn_features_idx = []
        for feature in feature_names:
            if feature == dom_number_label:
                self._dom_number_idx = feature_names.index(feature)
            elif feature == string_label:
                self._string_idx = feature_names.index(feature)
            else:
                self._cnn_features_idx.append(feature_names.index(feature))

    def forward(self, data: Data, data_feature_names: List[str]) -> Data:
        """Scatter pixel rows into IceCube-86 image tensor(s)."""
        # Initialize output arrays
        if self._include_main_array:
            main_arr = torch.zeros(
                (self._nb_cnn_features, 10, 10, 60),
                dtype=self._dtype,
            )
        if self._include_upper_dc:
            upper_dc_arr = torch.zeros(
                (self._nb_cnn_features, 8, 10),
                dtype=self._dtype,
            )
        if self._include_lower_dc:
            lower_dc_arr = torch.zeros(
                (self._nb_cnn_features, 8, 50),
                dtype=self._dtype,
            )

        # data.x is a tensor with shape (N, F) where N is the number of
        # pixels (DOMs) and F is the number of features. Each row
        # represents a single pixel.
        x = data.x

        string_dom_number = x[
            :, [self._string_idx, self._dom_number_idx]
        ].int()
        batch_row_features = x[:, self._cnn_features_idx]

        # Look up the pixel position in each sub-image for every (string,
        # dom_number) pair. Columns are referenced via the configurable
        # `string_label`/`dom_number_label` and the internal axis-column
        # constants so the lookup does not depend on free-form strings.
        match_indices = self._mapping.loc[
            zip(*string_dom_number.t().tolist())
        ][
            [
                self._string_label,
                self._dom_number_label,
                MAT_AX0_COL,
                MAT_AX1_COL,
                MAT_AX2_COL,
            ]
        ].values.astype(
            int
        )

        # Copy CNN features to the appropriate arrays
        for i, row in enumerate(match_indices):
            # Select appropriate array and indexing
            if row[0] < 79:  # Main Array
                if self._include_main_array:
                    main_arr[
                        :,
                        row[2],  # mat_ax0
                        row[3],  # mat_ax1
                        row[4],  # mat_ax2
                    ] = batch_row_features[i]

            elif row[1] < 11:  # Upper DeepCore
                if self._include_upper_dc:
                    upper_dc_arr[
                        :,
                        row[2],  # mat_ax0
                        row[3],  # mat_ax1
                    ] = batch_row_features[i]

            else:  # Lower DeepCore
                if self._include_lower_dc:
                    lower_dc_arr[
                        :,
                        row[2],  # mat_ax0
                        row[3],  # mat_ax1
                    ] = batch_row_features[i]

        # unsqueeze to add dimension for batching
        # with collate_fn Batch.from_data_list
        ret: List[torch.Tensor] = []
        if self._include_main_array:
            ret.append(main_arr.unsqueeze(0))
        if self._include_upper_dc:
            ret.append(upper_dc_arr.unsqueeze(0))
        if self._include_lower_dc:
            ret.append(lower_dc_arr.unsqueeze(0))

        # Set list of images as data.x
        data.x = ret
        return data

    def _set_image_feature_names(self, input_feature_names: List[str]) -> None:
        """Set the final output feature names."""
        # string and dom_number are only used for mapping
        # and will not be included in the output features.
        self.image_feature_names = [
            infeature
            for infeature in input_feature_names
            if infeature not in [self._string_label, self._dom_number_label]
        ]

    @property
    def shape(
        self,
    ) -> List[List[int]]:
        """Return the shape of the output images as a list of tuples."""
        ret = []
        if self._include_main_array:
            ret.append([self._nb_cnn_features, 10, 10, 60])
        if self._include_upper_dc:
            ret.append([self._nb_cnn_features, 1, 8, 10])
        if self._include_lower_dc:
            ret.append([self._nb_cnn_features, 1, 8, 50])
        return ret


class ExamplePrometheusGridDefinition(GridDefinition):
    """Example single-image grid for Prometheus-style layouts."""

    def __init__(
        self,
        detector: Detector,
        dtype: torch.dtype,
        pixel_feature_names: List[str],
        string_label: str = "sensor_string_id",
        sensor_number_label: str = "sensor_id",
    ) -> None:
        """Construct grid.

        Args:
            detector: Typically ``ORCA150`` in the example scripts.
            dtype: data type used for node features. e.g. ´torch.float´
            string_label: Name of the feature corresponding
                to the sensor string number.
            sensor_number_label: Name of the feature corresponding
                to the sensor number
            pixel_feature_names: Names of each column in expected input data
                that will be built into a image.

        Raises:
            ValueError: If no array type is included.

        NOTE: Expects input data to be sensors with aggregated features.
        """
        self._dtype = dtype
        self._string_label = string_label
        self._sensor_number_label = sensor_number_label
        self._pixel_feature_names = pixel_feature_names

        self._set_indices(
            pixel_feature_names, sensor_number_label, string_label
        )

        self._nb_cnn_features = (
            len(pixel_feature_names) - 2
        )  # 2 for string and sensor number

        # Hand-crafted layout for the example ORCA image; stored as an
        # embedded table (was ``prometheus_CNN_mapping.parquet``).
        df = pd.DataFrame(
            PROMETHEUS_CNN_MAPPING_ROWS,
            columns=[
                "sensor_string_id",
                "sensor_id",
                MAT_AX0_COL,
                MAT_AX1_COL,
                MAT_AX2_COL,
            ],
        )
        df.rename(
            columns={
                "sensor_string_id": string_label,
                "sensor_id": sensor_number_label,
            },
            inplace=True,
        )
        df.sort_values(
            by=[string_label, sensor_number_label],
            ascending=[True, True],
            inplace=True,
        )

        df.set_index(
            [string_label, sensor_number_label],
            inplace=True,
            drop=False,
        )

        self._mapping = df
        super().__init__(
            detector=detector, pixel_feature_names=pixel_feature_names
        )

    @property
    def map_pixels_by(self) -> List[str]:
        """String and sensor identifiers used for voxel lookup."""
        return [self._string_label, self._sensor_number_label]

    def mappings(self) -> List[pd.DataFrame]:
        """Return the embedded lookup table for the example layout."""
        return [self._mapping]

    def _set_indices(
        self,
        feature_names: List[str],
        sensor_number_label: str,
        string_label: str,
    ) -> None:
        """Set the indices for the features."""
        self._cnn_features_idx = []
        for feature in feature_names:
            if feature == sensor_number_label:
                self._sensor_number_idx = feature_names.index(feature)
            elif feature == string_label:
                self._string_idx = feature_names.index(feature)
            else:
                self._cnn_features_idx.append(feature_names.index(feature))

    def forward(self, data: Data, data_feature_names: List[str]) -> Data:
        """Scatter pixel rows into the example 3D image tensor."""
        # Initialize output arrays
        image_tensor = torch.zeros(
            (self._nb_cnn_features, 8, 9, 22),
            dtype=self._dtype,
        )

        # data.x is expected to be a tensor with shape (N, F)
        # where N is the number of nodes and F is the number of features.
        x = data.x

        # Direct coordinate and feature extraction
        string_sensor_number = x[
            :, [self._string_idx, self._sensor_number_idx]
        ].int()
        batch_row_features = x[:, self._cnn_features_idx]

        # Look up the pixel position in the image for every (string,
        # sensor_id) pair. Column references go through the configurable
        # labels and the internal axis-column constants so this method
        # does not rely on hard-coded column names from the data file.
        match_indices = self._mapping.loc[
            zip(*string_sensor_number.t().tolist())
        ][
            [
                self._string_label,
                self._sensor_number_label,
                MAT_AX0_COL,
                MAT_AX1_COL,
                MAT_AX2_COL,
            ]
        ].values.astype(
            int
        )

        # Copy CNN features to the appropriate arrays
        for i, row in enumerate(match_indices):
            # Select appropriate array and indexing
            image_tensor[
                :,
                row[2],  # mat_ax0
                row[3],  # mat_ax1
                row[4],  # mat_ax2
            ] = batch_row_features[i]

        # unsqueeze to add dimension for batching
        # with collate_fn Batch.from_data_list
        ret: List[torch.Tensor] = [image_tensor.unsqueeze(0)]

        # Set list of images as data.x
        data.x = ret
        return data

    def _set_image_feature_names(self, input_feature_names: List[str]) -> None:
        """Set the final output feature names."""
        # string and sensor_number are only used for mapping
        # and will not be included in the output features.
        self.image_feature_names = [
            infeature
            for infeature in input_feature_names
            if infeature not in [self._string_label, self._sensor_number_label]
        ]

    @property
    def shape(
        self,
    ) -> List[List[int]]:
        """Return the shape of the output images as a list of tuples."""
        return [[self._nb_cnn_features, 8, 9, 22]]
