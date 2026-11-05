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


def _config_paths_with_state_dict() -> list:
    """Return pretrained configs shipped alongside a state dict."""
    return [
        path
        for path in _config_paths()
        if path.endswith("_config.yml")
        and os.path.exists(path.replace("_config.yml", "_state_dict.pth"))
    ]


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


@pytest.mark.parametrize(
    "config_path", _config_paths_with_state_dict(), ids=_config_id
)
def test_pretrained_state_dict_loads(config_path: str) -> None:
    """Test that shipped weights load into their model without key mismatch.

    Only applies to models whose state dict is committed next to the
    config; a strict load verifies the weights and the current
    architecture still agree exactly.
    """
    config = ModelConfig.load(config_path)
    assert isinstance(config, ModelConfig)
    model = Model.from_config(config, trust=True)

    state_dict_path = config_path.replace("_config.yml", "_state_dict.pth")
    model.load_state_dict(state_dict_path)
