"""Unit tests for the pretrained models shipped with GraphNeT."""

import glob
import os

import pytest

from graphnet.constants import PRETRAINED_MODEL_DIR
from graphnet.models import Model
from graphnet.utilities.config import ModelConfig


def _config_paths() -> list:
    """Return every pretrained model config shipped in the repository."""
    return sorted(
        glob.glob(
            os.path.join(PRETRAINED_MODEL_DIR, "**", "*.yml"), recursive=True
        )
    )


def _config_id(path: str) -> str:
    """Return a readable test id relative to the pretrained model dir."""
    return os.path.relpath(path, PRETRAINED_MODEL_DIR)


@pytest.mark.parametrize("config_path", _config_paths(), ids=_config_id)
def test_pretrained_config_builds(config_path: str) -> None:
    """Test that every shipped pretrained config constructs a model.

    Guards the committed configs against silent rot when a constructor
    argument or class name changes elsewhere in the library.
    """
    config = ModelConfig.load(config_path)
    assert isinstance(config, ModelConfig)
    model = Model.from_config(config, trust=True)
    assert isinstance(model, Model)
