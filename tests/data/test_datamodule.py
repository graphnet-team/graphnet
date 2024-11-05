"""Unit tests for DataModule."""

from copy import deepcopy
import os
from typing import List, Any, Dict, Tuple
import pandas as pd
import sqlite3
import pytest
from glob import glob
from torch.utils.data import SequentialSampler
import numpy as np

from graphnet.constants import EXAMPLE_DATA_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.models.detector import IceCubeDeepCore
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.training.utils import save_selection


def get_dataset_size() -> int:
    """Return number of events in dataset."""
    path = f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"
    with sqlite3.connect(path) as conn:
        query = "select event_no from mc_truth"
        return pd.read_sql(query, conn).shape[0]


def extract_dataset_indices(
    file_path: str, dataset_kwargs: Dict[str, Any]
) -> List[int]:
    """Extract all available event ids."""
    if file_path.endswith(".db"):
        with sqlite3.connect(file_path) as conn:
            query = f'SELECT event_no FROM {dataset_kwargs["truth_table"]}'
            selection = (
                pd.read_sql(query, conn)["event_no"].to_numpy().tolist()
            )
    elif "merged" in file_path:
        files = glob(
            os.path.join(file_path, dataset_kwargs["truth_table"], "*.parquet")
        )
        selection = np.arange(0, len(files)).tolist()
    else:
        raise AssertionError(
            f"File extension not accepted: {file_path.split('.')[-1]}"
        )
    return selection


@pytest.fixture
def dataset_ref(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """Return the dataset reference."""
    return request.param


@pytest.fixture
def dataset_setup(dataset_ref: pytest.FixtureRequest) -> tuple:
    """Set up the dataset for testing.

    Args:
        dataset_ref: The dataset reference.

    Returns:
        A tuple with the dataset reference,
        dataset kwargs, and dataloader kwargs.
    """
    # Grab public dataset paths
    data_path = (
        f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"
        if dataset_ref is SQLiteDataset
        else f"{EXAMPLE_DATA_DIR}/parquet/prometheus/merged"
    )

    # Setup basic inputs; can be altered by individual tests
    graph_definition = KNNGraph(
        detector=IceCubeDeepCore(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.DEEPCORE,
    )

    dataset_kwargs = {
        "truth_table": "mc_truth",
        "pulsemaps": "total",
        "truth": TRUTH.PROMETHEUS,
        "features": FEATURES.PROMETHEUS,
        "path": data_path,
        "graph_definition": graph_definition,
    }

    dataloader_kwargs = {"batch_size": 2, "num_workers": 1, "shuffle": True}

    return dataset_ref, dataset_kwargs, dataloader_kwargs


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


@pytest.mark.parametrize(
    "dataset_ref", [SQLiteDataset, ParquetDataset], indirect=True
)
def test_single_dataset_without_selections(
    dataset_setup: Tuple[Any, Dict[str, Any], Dict[str, int]]
) -> None:
    """Verify GraphNeTDataModule behavior when no test selection is provided.

    Args:
        dataset_setup: Tuple with dataset reference,
        dataset arguments, and dataloader arguments.

    Raises:
        Exception: If the test dataloader is accessed
        without providing a test selection.
    """
    dataset_ref, dataset_kwargs, dataloader_kwargs = dataset_setup

    # Only training_dataloader args
    # Default values should be assigned to validation dataloader
    dm = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
    )

    train_dataloader = dm.train_dataloader
    val_dataloader = dm.val_dataloader

    with pytest.raises(Exception):
        # should fail because we provided no test selection
        test_dataloader = dm.test_dataloader  # noqa
    # validation loader should have shuffle = False by default
    assert isinstance(val_dataloader.sampler, SequentialSampler)
    # Should have identical batch_size
    assert val_dataloader.batch_size == train_dataloader.batch_size
    # Training dataloader should contain more batches
    assert len(train_dataloader) > len(val_dataloader)


@pytest.mark.parametrize(
    "dataset_ref", [SQLiteDataset, ParquetDataset], indirect=True
)
def test_single_dataset_with_selections(
    dataset_setup: Tuple[Any, Dict[str, Any], Dict[str, int]]
) -> None:
    """Test that selection functionality of DataModule behaves as expected.

    Args:
        dataset_setup (Tuple[Any, Dict[str, Any], Dict[str, int]]): A tuple
        containing the dataset reference, dataset arguments,
        and dataloader arguments.

    Returns:
        None
    """
    dataset_ref, dataset_kwargs, dataloader_kwargs = dataset_setup
    # extract all events
    file_path = dataset_kwargs["path"]
    selection = extract_dataset_indices(
        file_path=file_path, dataset_kwargs=dataset_kwargs
    )

    test_selection = selection[0:5]
    train_val_selection = selection[5:]

    # Only training_dataloader args
    # Default values should be assigned to validation dataloader
    dm = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
        selection=train_val_selection,
        test_selection=test_selection,
    )

    train_dataloader = dm.train_dataloader
    val_dataloader = dm.val_dataloader
    test_dataloader = dm.test_dataloader

    # Check that the training and validation dataloader contains
    # the same number of events as was given in the selection.
    if isinstance(dataset_ref, SQLiteDataset):
        a = len(train_dataloader.dataset) + len(val_dataloader.dataset)
        assert a == len(train_val_selection)  # type: ignore
        assert len(test_dataloader.dataset) == len(
            test_selection
        )  # noqa: E501  # type: ignore
    elif isinstance(dataset_ref, ParquetDataset):
        # Parquet dataset selection is batches not events
        a = train_dataloader.dataset._indices + val_dataloader.dataset._indices
        assert a == len(train_val_selection)
        assert len(test_dataloader.dataset._indices) == len(test_selection)

    # Training dataloader should have more batches
    assert len(train_dataloader) > len(val_dataloader)

    # validation loader should have shuffle = False by default
    assert isinstance(val_dataloader.sampler, SequentialSampler)
    # test loader should have shuffle = False by default
    assert isinstance(test_dataloader.sampler, SequentialSampler)


@pytest.mark.parametrize(
    "dataset_ref", [SQLiteDataset, ParquetDataset], indirect=True
)
def test_dataloader_args(
    dataset_setup: Tuple[Any, Dict[str, Any], Dict[str, int]]
) -> None:
    """Test that arguments to dataloaders are propagated correctly.

    Args:
        dataset_setup (Tuple[Any, Dict[str, Any], Dict[str, int]]): A tuple
        containing the dataset reference, dataset keyword arguments,
        and dataloader keyword arguments.

    Returns:
        None
    """
    dataset_ref, dataset_kwargs, dataloader_kwargs = dataset_setup
    val_dataloader_kwargs = deepcopy(dataloader_kwargs)
    test_dataloader_kwargs = deepcopy(dataloader_kwargs)

    # Setting batch sizes to different values
    val_dataloader_kwargs["batch_size"] = 1
    test_dataloader_kwargs["batch_size"] = 2
    dataloader_kwargs["batch_size"] = 3

    dm = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
        validation_dataloader_kwargs=val_dataloader_kwargs,
        test_dataloader_kwargs=test_dataloader_kwargs,
    )

    # Check that the resulting dataloaders have the right batch sizes
    assert dm.train_dataloader.batch_size == dataloader_kwargs["batch_size"]
    assert dm.val_dataloader.batch_size == val_dataloader_kwargs["batch_size"]
    assert (
        dm.test_dataloader.batch_size == test_dataloader_kwargs["batch_size"]
    )


@pytest.mark.parametrize(
    "dataset_ref", [SQLiteDataset, ParquetDataset], indirect=True
)
def test_ensemble_dataset_without_selections(
    dataset_setup: Tuple[Any, Dict[str, Any], Dict[str, int]]
) -> None:
    """Test ensemble dataset functionality without selections.

    Args:
        dataset_setup (Tuple[Any, Dict[str, Any], Dict[str, int]]): A tuple
        containing the dataset reference, dataset keyword arguments,
        and dataloader keyword arguments.

    Returns:
        None
    """
    # Make dataloaders from single dataset
    dataset_ref, dataset_kwargs, dataloader_kwargs = dataset_setup
    dm_single = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=deepcopy(dataset_kwargs),
        train_dataloader_kwargs=dataloader_kwargs,
    )

    # Copy dataset path twice; mimic ensemble dataset behavior
    ensemble_dataset_kwargs = deepcopy(dataset_kwargs)
    dataset_path = ensemble_dataset_kwargs["path"]
    ensemble_dataset_kwargs["path"] = [dataset_path, dataset_path]

    # Create dataloaders from multiple datasets
    dm_ensemble = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=ensemble_dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
    )

    # Test that the ensemble dataloaders contain more batches
    assert len(dm_single.train_dataloader) < len(dm_ensemble.train_dataloader)
    assert len(dm_single.val_dataloader) < len(dm_ensemble.val_dataloader)


@pytest.mark.parametrize("dataset_ref", [SQLiteDataset, ParquetDataset])
def test_ensemble_dataset_with_selections(
    dataset_setup: Tuple[Any, Dict[str, Any], Dict[str, int]]
) -> None:
    """Test ensemble dataset functionality with selections.

    Args:
        dataset_setup (Tuple[Any, Dict[str, Any], Dict[str, int]]): A tuple
        containing the dataset reference, dataset keyword arguments,
        and dataloader keyword arguments.

    Returns:
        None
    """
    # extract all events
    dataset_ref, dataset_kwargs, dataloader_kwargs = dataset_setup
    file_path = dataset_kwargs["path"]
    selection = extract_dataset_indices(
        file_path=file_path, dataset_kwargs=dataset_kwargs
    )

    # Copy dataset path twice; mimic ensemble dataset behavior
    ensemble_dataset_kwargs = deepcopy(dataset_kwargs)
    dataset_path = ensemble_dataset_kwargs["path"]
    ensemble_dataset_kwargs["path"] = [dataset_path, dataset_path]

    # pass two datasets but only one selection; should fail:
    with pytest.raises(Exception):
        _ = GraphNeTDataModule(
            dataset_reference=dataset_ref,
            dataset_args=ensemble_dataset_kwargs,
            train_dataloader_kwargs=dataloader_kwargs,
            selection=selection,
        )

    # Pass two datasets and two selections; should work:
    selection_1 = selection[0:5]
    selection_2 = selection[5:]
    dm = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=ensemble_dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
        selection=[selection_1, selection_2],
    )

    # Check that the number of events in train/val match
    a = len(dm.train_dataloader.dataset)
    b = len(dm.val_dataloader.dataset)
    n_events = a + b  # type: ignore

    assert n_events == get_dataset_size()

    # Pass two datasets, two selections and two test selections; should work
    dm2 = GraphNeTDataModule(
        dataset_reference=dataset_ref,
        dataset_args=ensemble_dataset_kwargs,
        train_dataloader_kwargs=dataloader_kwargs,
        selection=[selection, selection],
        test_selection=[selection_1, selection_2],
    )

    n_events = len(dm2.test_dataloader.dataset)  # type: ignore

    assert n_events == get_dataset_size()
