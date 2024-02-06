"""Unit tests for DataModule."""

from typing import Union, Dict, Any, List

import os
import pandas as pd
import pytest
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.training.utils import save_selection


@pytest.fixture
def selection() -> List[int]:
    """Return a selection."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def file_path(tmpdir: str) -> str:
    """Return a file path."""
    return os.path.join(tmpdir, "selection.csv")


def test_save_selection(selection: List[int], file_path: str) -> None:
    """Test `save_selection` function."""
    save_selection(selection, file_path)

    assert os.path.exists(file_path)

    with open(file_path, "r") as f:
        content = f.read()
        assert content.strip() == "1,2,3,4,5"
