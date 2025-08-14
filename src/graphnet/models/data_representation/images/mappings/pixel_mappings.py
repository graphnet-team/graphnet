"""Classes for mapping pixel data to images."""

from abc import abstractmethod
from typing import List
from torch_geometric.data import Data
import torch
import pandas as pd
import numpy as np

from graphnet.models import Model
from graphnet.constants import IC86_CNN_MAPPING, PROMETHEUS_CNN_MAPPING


class PixelMapping(Model):
    """Abstract class for mapping pixel data to images."""

    def __init__(
        self,
        pixel_feature_names: List[str],
    ) -> None:
        """Construct `PixelMapping`."""
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._set_image_feature_names(pixel_feature_names)

    @abstractmethod
    def forward(self, data: Data, data_feature_names: List[str]) -> Data:
        """Map pixel data to images.

        Args:
            data: The input data containing pixel features.
            data_feature_names: Names of each column in expected input data
                that will be built into a image.

        Returns:
            Data: The output data with images as features.
        NOTE: The output data.x should be a list of tensors,
            where each tensor corresponds to an image.

        Make sure to add a batch dimension to the tensors. E.g a picture
        with dimensions CxHxW = 10x64x64 should be returned as
        1x10x64x64.
        """
        raise NotImplementedError

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


class IC86PixelMapping(PixelMapping):
    """Mapping for the IceCube86.

    This mapping is based on the CNN mapping used
    in the multiple IceCube86 analysis.
    For further details see: https://arxiv.org/abs/2101.11589
    """

    def __init__(
        self,
        dtype: torch.dtype,
        pixel_feature_names: List[str],
        string_label: str = "string",
        dom_number_label: str = "dom_number",
        include_main_array: bool = True,
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
    ):
        """Construct `IC86PixelMapping`.

        Args:
            dtype: data type used for node features. e.g. ´torch.float´
            string_label: Name of the feature corresponding
                to the DOM string number. Values Integers betweem 1 - 86
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

        self._set_indeces(pixel_feature_names, dom_number_label, string_label)

        self._nb_cnn_features = (
            len(pixel_feature_names) - 2
        )  # 2 for string and dom_number

        self._include_main_array = include_main_array
        self._include_lower_dc = include_lower_dc
        self._include_upper_dc = include_upper_dc

        # read mapping from parquet file
        df = pd.read_parquet(IC86_CNN_MAPPING)
        df.sort_values(
            by=["string", "dom_number"],
            ascending=[True, True],
            inplace=True,
        )

        # Set the index to string and dom_number for faster lookup
        df.set_index(
            ["string", "dom_number"],
            inplace=True,
            drop=False,
        )

        self._mapping = df
        super().__init__(pixel_feature_names=pixel_feature_names)

    def _set_indeces(
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
        """Map pixel data to images."""
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

        # data.x is expected to be a tensor with shape (N, F)
        # where N is the number of nodes and F is the number of features.
        x = data.x

        # Direct coordinate and feature extraction
        string_dom_number = x[
            :, [self._string_idx, self._dom_number_idx]
        ].int()
        batch_row_features = x[:, self._cnn_features_idx]

        # look up the mapping for string and dom_number
        match_indices = self._mapping.loc[
            zip(*string_dom_number.t().tolist())
        ][
            ["string", "dom_number", "mat_ax0", "mat_ax1", "mat_ax2"]
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

        # unqueeze to add dimension for batching
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


class ExamplePrometheusMapping(PixelMapping):
    """Mapping for the Prometheus detector.

    This mapping is made for example purposes and is not optimized for
    any specific use case. There is no guarantee that this mapping will
    work with all Prometheus data.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        pixel_feature_names: List[str],
        string_label: str = "sensor_string_id",
        sensor_number_label: str = "sensor_id",
    ):
        """Construct `ExamplePrometheusMapping`.

        Args:
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

        self._set_indeces(
            pixel_feature_names, sensor_number_label, string_label
        )

        self._nb_cnn_features = (
            len(pixel_feature_names) - 2
        )  # 2 for string and sensor number

        # read mapping from parquet file
        df = pd.read_parquet(PROMETHEUS_CNN_MAPPING)
        df.sort_values(
            by=["sensor_string_id", "sensor_id"],
            ascending=[True, True],
            inplace=True,
        )

        # Set the index to string and sensor_number for faster lookup
        df.set_index(
            ["sensor_string_id", "sensor_id"],
            inplace=True,
            drop=False,
        )

        self._mapping = df
        super().__init__(pixel_feature_names=pixel_feature_names)

    def _set_indeces(
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
        """Map pixel data to images."""
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

        # look up the mapping for string and sensor_number
        match_indices = self._mapping.loc[
            zip(*string_sensor_number.t().tolist())
        ][
            ["sensor_string_id", "sensor_id", "mat_ax0", "mat_ax1", "mat_ax2"]
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

        # unqueeze to add dimension for batching
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
