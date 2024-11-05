"""Unit tests for DataConverter and Dataset classes."""

import os

import pandas as pd
import pytest
import sqlite3
import torch

import graphnet.constants
from graphnet.constants import TEST_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataconverter import DataConverter
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
)
from graphnet.data.extractors.internal import ParquetExtractor
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.dataset import ParquetDataset, SQLiteDataset
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.detector import IceCubeDeepCore

# Global variable(s)
TEST_DATA_DIR = os.path.join(
    graphnet.constants.TEST_DATA_DIR, "i3", "oscNext_genie_level7_v02"
)
FILE_NAME = "oscNext_genie_level7_v02_first_5_frames"
GCD_FILE = (
    "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
)


# Utility method(s)
def get_file_path(backend: str, table: str = "") -> str:
    """Return the path to the output file for `backend`."""
    suffix = {
        "sqlite": ".db",
        "parquet": ".parquet",
    }[backend]
    if backend == "sqlite":
        path = os.path.join(TEST_OUTPUT_DIR, backend, FILE_NAME + suffix)
    elif backend == "parquet":
        path = os.path.join(
            TEST_OUTPUT_DIR, backend, table, FILE_NAME + f"_{table}" + suffix
        )
    return path


@pytest.mark.order(1)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataconverter(
    backend: str, test_data_dir: str = TEST_DATA_DIR
) -> None:
    """Test the implementation of `DataConverter` for `backend`."""
    # Constructor DataConverter instance
    extractors = [
        I3TruthExtractor(),
        I3RetroExtractor(),
        I3FeatureExtractorIceCube86("SRTInIcePulses"),
    ]
    if backend == "sqlite":
        extractors.append(
            I3FeatureExtractorIceCube86("pulsemap_not_in_files_Pulses")
        )
    opt = dict(
        extractors=extractors,
        outdir=os.path.join(TEST_OUTPUT_DIR, backend),
        gcd_rescue=os.path.join(
            test_data_dir,
            GCD_FILE,
        ),
        workers=1,
    )

    converter: DataConverter
    if backend == "sqlite":
        converter = SQLiteDataConverter(**opt)  # type: ignore[arg-type]
    elif backend == "parquet":
        converter = ParquetDataConverter(**opt)  # type: ignore[arg-type]
    else:
        assert False, "Shouldn't reach here"

    # Perform conversion from I3 to `backend`
    converter(test_data_dir)
    converter.merge_files()

    # Check output
    if backend == "sqlite":
        path = get_file_path(backend)
        assert os.path.exists(path), path
    elif backend == "parquet":
        for extractor in extractors:
            table = extractor._extractor_name
            path = get_file_path(backend, table=table)
            assert os.path.exists(path), path


@pytest.mark.order(2)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataset(backend: str) -> None:
    """Test the implementation of `Dataset` for `backend`."""
    if backend == "sqlite":
        path = get_file_path(backend)
    elif backend == "parquet":
        path = os.path.join(TEST_OUTPUT_DIR, backend, "merged")
    graph_definition = KNNGraph(
        detector=IceCubeDeepCore(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.DEEPCORE,
    )

    # Constructor DataConverter instance
    opt = dict(
        path=path,
        pulsemaps="SRTInIcePulses",
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        graph_definition=graph_definition,
    )

    if backend == "sqlite":
        dataset = SQLiteDataset(**opt)  # type: ignore[arg-type]
    elif backend == "parquet":
        dataset = ParquetDataset(**opt)  # type: ignore[arg-type]
    else:
        assert False, "Shouldn't reach here"

    # Compare to expectations
    expected_number_of_events = 5
    test_indices = list(range(expected_number_of_events))
    expected_numbers_of_pulses = [
        9,
        9,
        12,
        11,
        15,
    ]

    assert len(dataset) == expected_number_of_events

    for ix, expected_number_of_pulses in zip(
        test_indices, expected_numbers_of_pulses
    ):
        event = dataset[ix]
        assert event.x.size(dim=0) == expected_number_of_pulses
        assert event.x.size(dim=0) == event.n_pulses
        assert event.x.size(dim=1) == len(event.features)
        assert isinstance(opt["features"], list), print(opt["features"])
        assert len(event.features) == len(opt["features"])

    for k in range(len(dataset)):
        dataset.__getitem__(k)


@pytest.mark.order(3)
def test_parquet_to_sqlite_converter() -> None:
    """Test the implementation of `ParquetToSQLiteConverter`."""
    # Constructor ParquetToSQLiteConverter instance
    outdir = os.path.join(TEST_OUTPUT_DIR, "parquet_to_sqlite")
    converter = ParquetToSQLiteConverter(
        extractors=[
            ParquetExtractor(extractor_name="truth"),
            ParquetExtractor(extractor_name="SRTInIcePulses"),
        ],
        outdir=outdir,
        num_workers=1,
    )
    graph_definition = KNNGraph(
        detector=IceCubeDeepCore(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.DEEPCORE,
    )
    # Perform conversion from I3 to parquet
    converter(os.path.join(TEST_OUTPUT_DIR, "parquet"))
    converter.merge_files()

    # Check that output exists
    path = f"{outdir}/merged/merged.db"
    assert os.path.exists(path), path

    # Check that datasets agree
    opt = dict(
        pulsemaps="SRTInIcePulses",
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        graph_definition=graph_definition,
    )

    dataset_from_parquet = SQLiteDataset(path, **opt)  # type: ignore[arg-type]
    dataset = SQLiteDataset(
        get_file_path("sqlite"), **opt  # type: ignore[arg-type]
    )

    assert len(dataset_from_parquet) == len(dataset)
    for ix in range(len(dataset)):
        dataset_from_parquet[ix].x
        dataset[ix].x
        assert torch.allclose(dataset_from_parquet[ix].x, dataset[ix].x)


@pytest.mark.order(4)
@pytest.mark.parametrize("pulsemap", ["SRTInIcePulses"])
@pytest.mark.parametrize("event_no", [1])
def test_database_query_plan(pulsemap: str, event_no: int) -> None:
    """Test query plan agreement in original and parquet-converted database."""
    # Configure paths to databases to compare
    parquet_converted_database = os.path.join(
        TEST_OUTPUT_DIR, "parquet_to_sqlite", "merged", "merged.db"
    )
    sqlite_database = get_file_path("sqlite")

    # Get query plans
    query = f"""
    EXPLAIN QUERY PLAN
    SELECT * FROM {pulsemap}
    WHERE event_no={event_no}
    """
    with sqlite3.connect(sqlite_database) as conn:
        sqlite_plan = pd.read_sql(query, conn)

    with sqlite3.connect(parquet_converted_database) as conn:
        parquet_plan = pd.read_sql(query, conn)

    # Compare
    assert "USING INDEX event_no" in sqlite_plan["detail"].iloc[0]
    assert "USING INDEX event_no" in parquet_plan["detail"].iloc[0]

    assert (sqlite_plan["detail"] == parquet_plan["detail"]).all()
