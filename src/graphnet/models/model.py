from abc import ABC

from pytorch_lightning import LightningModule

from graphnet.utilities.logging import LoggerMixin


class Model(LightningModule, LoggerMixin, ABC):
    """Base class for all models in graphnet."""
