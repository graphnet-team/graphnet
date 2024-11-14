"""Base detector-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Dict, Callable, List, Optional

from torch_geometric.data import Data
import torch
import pandas as pd

from graphnet.models import Model
from graphnet.utilities.decorators import final


class Detector(Model):
    """Base class for all detector-specific read-ins in graphnet."""

    def __init__(
        self, replace_with_identity: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`.

        Args:
            replace_with_identity: A list of feature names from the
            feature_map that should be replaced with the identity
            function.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._replace_with_identity = replace_with_identity

    @abstractmethod
    def feature_map(self) -> Dict[str, Callable]:
        """List of features used/assumed by inheriting `Detector` objects."""

    @final
    def forward(  # type: ignore
        self, input_features: torch.tensor, input_feature_names: List[str]
    ) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""
        return self._standardize(input_features, input_feature_names)

    @property
    def geometry_table(self) -> pd.DataFrame:
        """Public get method for retrieving a `Detector`s geometry table."""
        if ~hasattr(self, "_geometry_table"):
            try:
                assert hasattr(self, "geometry_table_path")
            except AssertionError as e:
                self.error(
                    f"""{self.__class__.__name__} does not have class
                           variable `geometry_table_path` set."""
                )
                raise e
            self._geometry_table = pd.read_parquet(self.geometry_table_path)
        return self._geometry_table

    @property
    def string_index_name(self) -> str:
        """Public get method for retrieving the string index column name."""
        return self.string_id_column

    @property
    def sensor_position_names(self) -> List[str]:
        """Public get method for retrieving the xyz coordinate column names."""
        return self.xyz

    @property
    def sensor_index_name(self) -> str:
        """Public get method for retrieving the sensor id column name."""
        return self.sensor_id_column

    @final
    def _standardize(
        self, input_features: torch.tensor, input_feature_names: List[str]
    ) -> Data:
        feature_map = self.feature_map()
        if self._replace_with_identity is not None:
            for feature in self._replace_with_identity:
                feature_map[feature] = self._identity
        for idx, feature in enumerate(input_feature_names):
            try:
                input_features[:, idx] = feature_map[
                    feature
                ](  # noqa: E501 # type: ignore
                    input_features[:, idx]
                )
            except KeyError as e:
                self.warning(
                    f"""No Standardization function found for '{feature}'"""
                )
                raise e
        return input_features

    def _identity(self, x: torch.tensor) -> torch.tensor:
        """Apply no standardization to input."""
        return x
