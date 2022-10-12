from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

try:
    from typing import final
except ImportError:  # Python version < 3.8

    def final(f):  # Identity decorator
        return f


from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data


from graphnet.utilities.logging import LoggerMixin

if TYPE_CHECKING:
    # Avoid cyclic dependency
    from graphnet.models.config import ModelConfig


class Model(LightningModule, LoggerMixin, ABC):
    """Base class for all models in graphnet."""

    @abstractmethod
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""

    @final
    @property
    def config(self) -> "ModelConfig":
        """Configuration to re-create the model."""
        try:
            return self._config
        except AttributeError:
            self.logger.error(
                "ModelConfig was not set. "
                "Did you wrap the class constructor with `save_config`?"
            )
            raise

    @final
    def save_config(self, path: str):
        """Save ModelConfig to `path` as YAML file."""
        self.config.dump(path)

    @classmethod
    def from_config(
        cls,
        source: Union["ModelConfig", str],
        trust: bool = False,
    ) -> "Model":
        """Construct `Model` instance from `source` configuration.

        Arguments:
            trust (bool): Whether to trust the ModelConfig file enough to
                `eval(...)` any lambda function expressions contained.

        Raises:
            ValueError: If the ModelConfig contains lambda functions but
                `trust = False`.
        """
        from graphnet.models.config import ModelConfig

        if isinstance(source, str):
            source = ModelConfig.load(source)

        assert isinstance(source, ModelConfig)
        return source.construct_model(trust=trust)
