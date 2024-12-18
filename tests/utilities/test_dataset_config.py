"""Unit tests for `DatasetConfig` class."""

import json
import os.path
from typing import Dict

import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset

import graphnet
import graphnet.constants
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset
from graphnet.data.dataset import ParquetDataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.utilities.config import DatasetConfig
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.graphs.nodes import NodesAsPulses


CONFIG_PATHS = {
    "parquet": "/tmp/test_dataset_parquet.yml",
    "sqlite": "/tmp/test_dataset_sqlite.yml",
}

graph_definition = KNNGraph(
    detector=IceCubeDeepCore(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
    input_feature_names=FEATURES.DEEPCORE,
)


@pytest.mark.order(1)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_save_load_reconstruct(backend: str) -> None:
    """Test saving, loading, and reconstructing dataset."""
    # Arrange
    config_path = CONFIG_PATHS[backend]
    dataset_name = "oscNext_genie_level7_v02"
    file_name = "oscNext_genie_level7_v02_first_5_frames"
    suffix = {
        "sqlite": "db",
        "parquet": "parquet",
    }[backend]

    if backend == "sqlite":
        path = os.path.join(
            graphnet.constants.TEST_DATA_DIR,
            backend,
            dataset_name,
            f"{file_name}.{suffix}",
        )
    elif backend == "parquet":
        path = os.path.join(
            graphnet.constants.TEST_DATA_DIR,
            backend,
            dataset_name,
            "merged",
        )
    # Constructor DataConverter instance
    opt = dict(
        path=path,
        pulsemaps="SRTInIcePulses",
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        selection="event_no % 5 > 0",
        graph_definition=graph_definition,
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
    assert os.path.exists(config_path)

    # Load config from file
    loaded_config = DatasetConfig.load(config_path)
    assert isinstance(loaded_config, DatasetConfig)

    # Reconstruct dataset
    constructed_dataset = Dataset.from_config(loaded_config)
    assert isinstance(constructed_dataset, Dataset)
    assert constructed_dataset.config == dataset.config
    assert len(constructed_dataset) == len(dataset)
    nb_test_events = min(5, len(constructed_dataset))
    for ix in range(nb_test_events):
        assert torch.all(constructed_dataset[ix].x == dataset[ix].x)


@pytest.mark.order(2)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_dict_selection(backend: str) -> None:
    """Test constructing Dataset with dictionary of selections."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    # Construct multiple datasets
    config = DatasetConfig.load(config_path)
    config.selection = {
        "train": "event_no % 5 > 0",
        "test": "event_no % 5 == 0",
    }

    datasets: Dict[str, Dataset] = Dataset.from_config(config)
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


@pytest.mark.order(3)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_list_selection(backend: str) -> None:
    """Test constructing Dataset with list of selections."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    # Construct multiple datasets
    config = DatasetConfig.load(config_path)
    config.selection = ["event_no % 5 == 0", "event_no % 5 >= 3"]

    dataset: ConcatDataset = Dataset.from_config(config)
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 3
    for event in dataset:
        assert (event["event_no"] % 5) not in (1, 2)


@pytest.mark.order(3)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_dict_of_list_selection(backend: str) -> None:
    """Test constructing Dataset with dictionary of lists of selections."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    # Construct multiple datasets
    config = DatasetConfig.load(config_path)
    config.selection = {
        "train": ["event_no % 5 == 0"],
        "test": ["event_no % 5 == 1", "event_no % 5 == 2"],
    }

    datasets: Dict[str, ConcatDataset] = Dataset.from_config(config)
    assert isinstance(datasets, dict)
    for dataset in datasets.values():
        assert isinstance(dataset, ConcatDataset)

    # Check that event counts match expectation
    assert len(datasets) == 2
    assert len(datasets["train"]) == 1
    assert len(datasets["test"]) == 2


@pytest.mark.order(4)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_functions(backend: str) -> None:
    """Test constructing Dataset with selections containing functions."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    # Construct multiple datasets
    config = DatasetConfig.load(config_path)
    config.selection = {
        "nu_mu": "pid == 14 and 10**(energy) > 100",
        "nuhat_mu": "pid == -14 and 10**(energy) > 100",
        "nu_nuhat_mu": "abs(pid) == 14 and 10**(energy) > 100",
        "inverse": "abs(pid) == 14 and 10**(energy) <= 100",
    }

    datasets: Dict[str, Dataset] = Dataset.from_config(config)

    # Check that event counts match expectation
    assert len(datasets["nu_mu"]) == 1
    assert len(datasets["nuhat_mu"]) == 2
    assert len(datasets["nu_nuhat_mu"]) == 3
    assert len(datasets["inverse"]) == 2

    # Check that selections work by making sure there is no overlap between
    # event_nos
    assert (
        len(
            set(datasets["nu_mu"]._indices).intersection(
                set(datasets["nuhat_mu"]._indices)
            )
        )
        == 0
    )
    assert set(datasets["nu_mu"]._indices).union(
        set(datasets["nuhat_mu"]._indices)
    ) == set(datasets["nu_nuhat_mu"]._indices)
    assert (
        len(
            set(datasets["nu_nuhat_mu"]._indices).intersection(
                set(datasets["inverse"]._indices)
            )
        )
        == 0
    )
    assert (
        len(
            set(datasets["nu_nuhat_mu"]._indices).union(
                set(datasets["inverse"]._indices)
            )
        )
        == 5
    )


@pytest.mark.order(5)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_dataset_config_files(backend: str) -> None:
    """Test constructing Dataset with selections containing functions."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    selection_file_csv = "/tmp/test_selection.csv"
    selection_file_json = "/tmp/test_selection.json"

    # Save selection(s) to file(s)
    indices_csv = [1, 2, 4]
    indices_json = [0, 3]
    df_selection = pd.DataFrame(data=indices_csv, columns=["event_no"])
    df_selection.to_csv(selection_file_csv)
    with open(selection_file_json, "w") as f:
        json.dump(indices_json, f)

    # Construct multiple datasets
    config = DatasetConfig.load(config_path)
    config.seed = 2
    config.selection = {
        "CSV": f"2 random events ~ {selection_file_csv}",
        "JSON": selection_file_json,
    }

    datasets: Dict[str, Dataset] = Dataset.from_config(config)

    # Check that event counts match expectation
    assert len(datasets["CSV"]) == 2
    assert len(datasets["JSON"]) == 2

    # Check that selections work by making sure there is no overlap between
    # event_nos
    assert (
        len(
            set(datasets["CSV"]._indices).intersection(
                set(datasets["JSON"]._indices)
            )
        )
        == 0
    )


@pytest.mark.order(6)
@pytest.mark.parametrize("backend", ["sqlite"])
def test_multiple_dataset_config_dict_selection(backend: str) -> None:
    """Test constructing Dataset with multiple data paths."""
    # Arrange
    config_path = CONFIG_PATHS[backend]

    # Single dataset
    config = DatasetConfig.load(config_path)
    dataset = Dataset.from_config(config)
    # Construct multiple datasets
    config_ensemble = DatasetConfig.load(config_path)
    config_ensemble.path = [config_ensemble.path, config_ensemble.path]

    ensemble_dataset = Dataset.from_config(config_ensemble)

    assert len(dataset) * 2 == len(ensemble_dataset)
