"""Tests for examples in 02_data."""
import runpy
import os
from graphnet.constants import GRAPHNET_ROOT_DIR

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/02_data")


def test_01_read_dataset() -> None:
    """Test for 01_read_dataset."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, "01_read_dataset.py"))


def test_02_plot_feature_distribution() -> None:
    """Test for 02_plot_feature_distribution."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "02_plot_feature_distributions.py")
    )


def test_03_convert_parquet_to_sqlite() -> None:
    """Test for 03_convert_parquet_to_sqlite."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "03_convert_parquet_to_sqlite.py")
    )


def test_04_ensemble_dataset() -> None:
    """Test for 04_ensemble_dataset."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, "04_ensemble_dataset.py"))
