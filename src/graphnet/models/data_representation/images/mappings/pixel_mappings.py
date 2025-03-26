"""Classes for mapping pixel data to images."""

from abc import abstractmethod
from typing import List
from torch_geometric.data import Data
import torch
import pandas as pd

from graphnet.models import Model
from graphnet.constants import IC86_CNN_MAPPING


class PixelMapping(Model):
    """Abstract class for mapping pixel data to images."""

    def __init__(
        self,
    ) -> None:
        """Construct `PixelMapping`."""
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def forward(self, data: Data, data_feature_names: List[str]) -> Data:
        """Map pixel data to images.

        Make sure to add a batch dimension to the output. E.g picture with
        dimensions CxHxW = 10x64x64 should be returned as 1x10x64x64.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_image_feature_names(self, input_feature_names: List[str]) -> None:
        """Set the final image feature names."""
        raise NotImplementedError


class IC86DNNMapping(PixelMapping):
    """Mapping for the IceCube86.

    This mapping is based on the CNN mapping used in the IceCube86 analysis.
    See: https://arxiv.org/abs/2101.11589
    """

    def __init__(
        self,
        dtype: torch.dtype,
        dom_pos_names: List[str],
        pixel_feature_names: List[str],
        include_lower_dc: bool = True,
        include_upper_dc: bool = True,
    ):
        """Construct `IC86MircoDNNMapping`.

        Args:
            dtype: data type used for node features. e.g. ´torch.float´
            dom_pos_names: Names of the DOM position features.
            pixel_feature_names: Names of each column in expected input data
                that will be built into a image.
            include_lower_dc: If True, the lower DeepCore will be included.
            include_upper_dc: If True, the upper DeepCore will be included.
        """
        super().__init__()
        self._dtype = dtype
        self._dom_pos_names = dom_pos_names
        self._pixel_feature_names = pixel_feature_names

        self._set_indeces(pixel_feature_names, dom_pos_names)

        self._nb_cnn_features = len(pixel_feature_names) - len(dom_pos_names)

        self._include_lower_dc = include_lower_dc
        self._include_upper_dc = include_upper_dc

        self._tensor_mapping = torch.tensor(
            pd.read_parquet(IC86_CNN_MAPPING).values,
            dtype=dtype,
        )

    def _set_indeces(
        self,
        feature_names: List[str],
        dom_pos_names: List[str],
    ) -> None:
        self._dom_pos_idx = []
        self._cnn_features_idx = []
        for feature in feature_names:
            if feature in dom_pos_names:
                self._dom_pos_idx.append(feature_names.index(feature))
            else:
                self._cnn_features_idx.append(feature_names.index(feature))

    def forward(
        self, data: Data, data_feature_names: List[str]
    ) -> List[torch.Tensor]:
        """Map pixel data to images."""
        # Initialize output arrays

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

        x = data.x

        # Direct coordinate and feature extraction
        batch_coords = x[:, self._dom_pos_idx]
        batch_row_features = x[:, self._cnn_features_idx]

        # Compute coordinate matches directly
        coord_matches = torch.all(
            torch.isclose(
                batch_coords.unsqueeze(1),
                self._tensor_mapping[:, :3].unsqueeze(0),
                rtol=1e-5,
            ),
            dim=-1,
        )

        # Find matching indices
        match_indices = coord_matches.nonzero(as_tuple=False)

        assert match_indices.numel() != 0

        # Process matches efficiently
        for match_row, geom_idx in match_indices:
            # Retrieve geometric information directly from tensor
            string_val = self._tensor_mapping[geom_idx, 6].item()
            dom_number = self._tensor_mapping[geom_idx, 7].item()

            # Select appropriate array and indexing
            if string_val < 79:  # Main Array
                main_arr[
                    :,
                    int(self._tensor_mapping[geom_idx, 3]),
                    int(self._tensor_mapping[geom_idx, 4]),
                    int(self._tensor_mapping[geom_idx, 5]),
                ] = batch_row_features[match_row]

            elif dom_number < 11:  # Upper DeepCore
                if self._include_upper_dc:
                    upper_dc_arr[
                        :,
                        int(self._tensor_mapping[geom_idx, 3]),
                        int(self._tensor_mapping[geom_idx, 4]),
                    ] = batch_row_features[match_row]

            else:  # Lower DeepCore
                if self._include_lower_dc:
                    lower_dc_arr[
                        :,
                        int(self._tensor_mapping[geom_idx, 3]),
                        int(self._tensor_mapping[geom_idx, 4]),
                    ] = batch_row_features[match_row]

        # unqueeze to add batch dimension
        ret = [main_arr.unsqueeze(0)]
        if self._include_upper_dc:
            ret.append(upper_dc_arr.unsqueeze(0))
        if self._include_lower_dc:
            ret.append(lower_dc_arr.unsqueeze(0))

        data.x = ret

        return data

    def _set_image_feature_names(self, input_feature_names: List[str]) -> None:
        """Set the final output feature names."""
        self.image_feature_names = [
            infeature
            for infeature in input_feature_names
            if infeature not in self._dom_pos_names
        ]
