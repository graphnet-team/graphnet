from abc import ABC, abstractmethod
from typing import Union

from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data

from graphnet.utilities.logging import LoggerMixin


class Model(LightningModule, LoggerMixin, ABC):
    """Base class for all models in graphnet."""

    @abstractmethod
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""
