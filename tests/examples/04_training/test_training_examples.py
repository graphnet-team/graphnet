"""Test for examples in 04_training."""

import runpy
import os
from glob import glob
import pytest

from graphnet.constants import GRAPHNET_ROOT_DIR

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/04_training")


examples = glob(EXAMPLE_PATH + "/*.py")


@pytest.mark.parametrize("example", examples)
def test_script_execution(example: str) -> None:
    """Test function that executes example."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, example), run_name="__main__")
