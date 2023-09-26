"""Test for examples in 04_training."""
import runpy
import os
from graphnet.constants import GRAPHNET_ROOT_DIR

EXAMPLE_PATH = os.path.join(GRAPHNET_ROOT_DIR, "examples/04_training")


def test_01_train_dynedge() -> None:
    """Test for 01_train_dynedge."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "01_train_dynedge.py"), run_name="__main__"
    )


def test_02_train_tito_model() -> None:
    """Test for 02_train_tito_model."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "02_train_tito_model.py"),
        run_name="__main__",
    )


def test_03_train_dynedge_from_config() -> None:
    """Test for 03_train_dynedge_from_config."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "03_train_dynedge_from_config.py"),
        run_name="__main__",
    )


def test_04_train_multiclassifier_from_configs() -> None:
    """Test for 04_train_multiclassifier_from_configs."""
    runpy.run_path(
        os.path.join(EXAMPLE_PATH, "04_train_multiclassifier_from_configs.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    test_01_train_dynedge()
    test_02_train_tito_model()
    test_03_train_dynedge_from_config()
    test_04_train_multiclassifier_from_configs()
