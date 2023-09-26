"""Test for examples in 01_icetray."""
import runpy
import os
from graphnet.constants import GRAPHNET_ROOT_DIR

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/01_icetray")


def test_01_convert_i3_files() -> None:
    """Test for 01_convert_i3_files."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, "01_convert_i3_files.py"))


def test_02_compare_sqlite_and_parquet() -> None:
    """Test for 02_compare_sqlite_and_parquet."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "02_compare_sqlite_and_parquet.py")
    )


def test_03_i3_deployer_example() -> None:
    """Test for 03_i3_deployer_example."""
    runpy.run_path(os.path.join(EXAMPLE_PATH, "03_i3_deployer_example.py"))


def test_04_i3_module_in_native_icetray_example() -> None:
    """Test for 04_i3_module_in_native_icetray_example."""
    runpy.run_path(
        os.path.join(
            EXAMPLE_PATH, "04_i3_module_in_naticve_icetray_example.py"
        )
    )
