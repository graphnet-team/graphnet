"""Class(es) for constructing training labels at runtime."""

from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import Logger


class Label(ABC, Logger):
    """Base `Label` class for producing labels from single `Data` instance."""

    def __init__(self, key: str):
        """Construct `Label`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
        """
        self._key = key

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, graph: Data) -> torch.tensor:
        """Label-specific implementation."""


class Direction(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self, azimuth_key: str = "azimuth", zenith_key: str = "zenith"
    ):
        """Construct `Direction`."""
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key="_".join([azimuth_key, zenith_key]))

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return torch.cat((x, y, z), dim=1)
