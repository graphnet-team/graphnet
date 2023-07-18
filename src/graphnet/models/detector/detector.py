"""Base detector-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Dict, Callable, List

from torch_geometric.data import Data
import torch

from graphnet.models import Model
from graphnet.utilities.decorators import final
from graphnet.utilities.config import save_model_config


class Detector(Model):
    """Base class for all detector-specific read-ins in graphnet."""

    @save_model_config
    def __init__(self) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @property
    @abstractmethod
    def feature_map(self) -> Dict[str, Callable]:
        """List of features used/assumed by inheriting `Detector` objects."""

    @final
    def forward(  # type: ignore
        self, node_features: torch.tensor, node_feature_names: List[str]
    ) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""
        return self._standardize(node_features, node_feature_names)

    @final
    def _standardize(
        self, node_features: torch.tensor, node_feature_names: List[str]
    ) -> Data:
        for idx, feature in enumerate(node_feature_names):
            try:
                node_features[:, idx] = self.feature_map()[feature](  # type: ignore
                    node_features[:, idx]
                )
            except KeyError as e:
                self.warning(
                    f"""No Standardization function found for '{feature}'"""
                )
                raise e
        return node_features
