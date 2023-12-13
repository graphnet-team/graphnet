"""Base class(es) for building models."""

from abc import ABC
import dill
import os.path
from typing import Any, Dict, List, Optional, Union

from pytorch_lightning import LightningModule
import torch
from torch_geometric.data import Data

from graphnet.utilities.deprecation_tools import rename_state_dict_entries
from graphnet.utilities.logging import Logger
from graphnet.utilities.config import (
    Configurable,
    ModelConfig,
    ModelConfigSaverABC,
)


class Model(
    Logger, Configurable, LightningModule, ABC, metaclass=ModelConfigSaverABC
):
    """Base class for all components in graphnet."""

    @staticmethod
    def _get_batch_size(data: List[Data]) -> int:
        return sum([torch.numel(torch.unique(d.batch)) for d in data])

    def save(self, path: str) -> None:
        """Save entire model to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix for model files."
            )
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.cpu(), path, pickle_module=dill)
        self.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load entire model from `path`."""
        return torch.load(path, pickle_module=dill)

    def save_state_dict(self, path: str) -> None:
        """Save model `state_dict` to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix for state_dict files."
            )
        torch.save(self.cpu().state_dict(), path)
        self.info(f"Model state_dict saved to {path}")

    def load_state_dict(
        self, path: Union[str, Dict], **kargs: Optional[Any]
    ) -> "Model":  # pylint: disable=arguments-differ
        """Load model `state_dict` from `path`."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path

        # DEPRECATION UTILITY: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647
        state_dict, state_dict_altered = rename_state_dict_entries(
            state_dict=state_dict, old_phrase="_gnn", new_phrase="backbone"
        )
        if state_dict_altered:
            self.warn(
                "DeprecationWarning: State dicts with `_gnn` entries will be deprecated in GraphNeT 2.0"
            )
        return super().load_state_dict(state_dict, **kargs)

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[ModelConfig, str],
        trust: bool = False,
        load_modules: Optional[List[str]] = None,
    ) -> "Model":
        """Construct `Model` instance from `source` configuration.

        Arguments:
            trust: Whether to trust the ModelConfig file enough to `eval(...)`
                any lambda function expressions contained.
            load_modules: List of modules used in the definition of the model
                which, as a consequence, need to be loaded into the global
                namespace. Defaults to loading `torch`.

        Raises:
            ValueError: If the ModelConfig contains lambda functions but
                `trust = False`.
        """
        if isinstance(source, str):
            source = ModelConfig.load(source)

        assert isinstance(
            source, ModelConfig
        ), f"Argument `source` of type ({type(source)}) is not a `ModelConfig"

        return source._construct_model(trust, load_modules)
