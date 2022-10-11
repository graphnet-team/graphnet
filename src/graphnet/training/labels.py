from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import LoggerMixin


class Label(ABC, LoggerMixin):
    """Base Label class for producing labels from single Data Object."""

    def __init__(self, key: str):
        """Base Label class for producing labels from single Data Object.

        Args:
            key (str): The name of the field in Data where the label will be stored. Ie. graph[key] = label
        """
        self._key = key

    @abstractmethod
    def __call__(self, graph: Data) -> torch.tensor:
        """Label-specific implementation"""


class Direction(Label):
    def __init__(
        self, azimuth_key: str = "azimuth", zenith_key: str = "zenith"
    ):
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

    def __call__(self, graph: Data) -> torch.tensor:
        x = torch.sin(graph[self._azimuth_key]) * torch.cos(
            graph[self._zenith_key]
        )
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        )
        z = torch.cos(graph[self._azimuth_key])
        return torch.cat([[x, y, z]], dtype=torch.float)
