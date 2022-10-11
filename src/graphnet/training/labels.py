from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import LoggerMixin


class Label(ABC, LoggerMixin):
    """Base Dataset class for producing labels from single Data Object."""

    def __init__(self, key: str):
        self._key = key

    @abstractmethod
    def __call__(self, graph: Data) -> Data:
        """Label-specific implementation"""


class DirectionReconstructionWithKappa(Label):
    def __init__(
        self, azimuth_key: str = "azimuth", zenith_key: str = "zenith"
    ):
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

    def __call__(self, graph: Data) -> Data:
        x = torch.sin(graph[self._azimuth_key]) * torch.cos(
            graph[self._zenith_key]
        )
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        )
        z = torch.cos(graph[self._azimuth_key])
        return torch.cat([[x, y, z]], dtype=torch.float)
