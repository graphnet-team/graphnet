"""Test for examples in 03_weights."""

import runpy
import os
from graphnet.constants import GRAPHNET_ROOT_DIR
from glob import glob
import pytest

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/03_weights")

examples = glob(EXAMPLE_PATH + "/*.py")


@pytest.mark.parametrize("example", examples)
def test_script_execution(example: str) -> None:
    """Test function that executes example."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, example))
