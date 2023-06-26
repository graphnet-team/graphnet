"""Base detector-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Dict, Callable

from torch_geometric.data import Data

from graphnet.models import Model
from graphnet.utilities.decorators import final


class Detector(Model):
    """Base class for all detector-specific read-ins in graphnet."""

    def __init__(self, feature_map: Dict[str, Callable]):
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self.feature_map = feature_map

    @final
    def forward(self, graph: Data) -> Data:
        """Pre-process graph `Data` features and build graph adjacency."""
        # Check(s)
        assert isinstance(graph, Data)
        return self._standardize(graph)

    @final
    def _standardize(self, graph: Data) -> Data:
        for feature, idx in graph.features:
            try:
                graph.x[:, idx] = self.feature_map[feature](graph.x[:, idx])
            except KeyError as e:
                self.warning(
                    f"""No Standardization function found for '{feature}'"""
                )
                raise e
        return graph
