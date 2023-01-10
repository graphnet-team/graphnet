"""Unit tests for `DatasetConfig` class."""

import os.path
from typing import Dict

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
        selection="event_no % 5 > 0",
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

    # Construct dataset
    constructed_dataset = Dataset.from_config(loaded_config)
    assert isinstance(constructed_dataset, Dataset)
    assert constructed_dataset.config == dataset.config
    assert len(constructed_dataset) == len(dataset)
    nb_test_events = min(5, len(constructed_dataset))
    for ix in range(nb_test_events):
        assert torch.all(constructed_dataset[ix].x == dataset[ix].x)

    # Construct multiple datasets
    dataset.config.selection = {
        "train": "event_no % 5 > 0",
        "test": "event_no % 5 == 0",
    }
    dataset.save_config(config_path)

    datasets: Dict[str, Dataset] = Dataset.from_config(config_path)
    assert isinstance(datasets, dict)
    assert len(datasets) == 2
    assert "train" in datasets and "test" in datasets

    # Check that selections work by making sure there is no overlap between
    # event_nos
    assert (
        len(
            set(datasets["train"]._indices).intersection(
                set(datasets["test"]._indices)
            )
        )
        == 0
    )


if __name__ == "__main__":
    test_dataset_config("sqlite")
