"""Unit tests for `DatasetConfig` class."""

import os.path

import pytest
import torch

import graphnet
import graphnet.constants
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset
from graphnet.data.parquet import ParquetDataset
from graphnet.data.sqlite import SQLiteDataset
from graphnet.utilities.config import DatasetConfig


@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataset_config(
    backend: str, config_path: str = "/tmp/simple_dataset.yml"
) -> None:
    """Test saving, loading, and reconstructing simple model."""
    # Arrange
    dataset_name = "oscNext_genie_level7_v02"
    file_name = "oscNext_genie_level7_v02_first_5_frames"
    suffix = {
        "sqlite": "db",
        "parquet": "parquet",
    }[backend]

    path = os.path.join(
        graphnet.constants.TEST_DATA_DIR,
        backend,
        dataset_name,
        f"{file_name}.{suffix}",
    )

    # Constructor DataConverter instance
    opt = dict(
        path=path,
        pulsemaps="SRTInIcePulses",
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
    )

    if backend == "sqlite":
        dataset = SQLiteDataset(**opt)  # type: ignore[arg-type]
    elif backend == "parquet":
        dataset = ParquetDataset(**opt)  # type: ignore[arg-type]
    else:
        assert False, "Shouldn't reach here"

    # Save config to file
    dataset.save_config(config_path.replace(".yml", ""))
    assert os.path.exists(config_path)
    dataset.save_config(config_path)

    # Load config from file
    loaded_config = DatasetConfig.load(config_path)
    assert isinstance(loaded_config, DatasetConfig)
    assert loaded_config == dataset.config

    # Construct model
    constructed_dataset_1 = Dataset.from_config(loaded_config)
    constructed_dataset_2 = loaded_config.construct_dataset()
    assert constructed_dataset_1.config == constructed_dataset_2.config
    assert len(constructed_dataset_1) == len(constructed_dataset_2)
    nb_test_events = min(5, len(constructed_dataset_1))
    for ix in range(nb_test_events):
        assert torch.all(
            constructed_dataset_1[ix].x == constructed_dataset_2[ix].x
        )
