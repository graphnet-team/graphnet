"""Unit tests for DataConverter and Dataset classes."""

import os

import numpy as np
import pytest

import graphnet.constants
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataconverter import DataConverter
from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
    I3GenericExtractor,
)
from graphnet.data.parquet import (
    ParquetDataset,
    ParquetDataConverter,
)
from graphnet.data.sqlite import (
    SQLiteDataset,
    SQLiteDataConverter,
)
from graphnet.data.sqlite.sqlite_dataconverter import (
    is_pulsemap_check,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import dataio  # pyright: reportMissingImports=false


# Global variable(s)
TEST_DATA_DIR = os.path.join(
    graphnet.constants.TEST_DATA_DIR, "i3", "oscNext_genie_level7_v02"
)
OUTPUT_DATA_DIR = os.path.join(graphnet.constants.TEST_DATA_DIR, "output")
FILE_NAME = "oscNext_genie_level7_v02_first_5_frames"
GCD_FILE = (
    "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
)


# Utility method(s)
def get_file_path(backend: str) -> str:
    """Return the path to the output file for `backend`."""
    suffix = {
        "sqlite": ".db",
        "parquet": ".parquet",
    }[backend]

    path = os.path.join(OUTPUT_DATA_DIR, FILE_NAME + suffix)
    return path


# Unit test(s)
def test_is_pulsemap_check() -> None:
    """Test behaviour of `is_pulsemap_check`."""
    assert is_pulsemap_check("SplitInIcePulses") is True
    assert is_pulsemap_check("SRTInIcePulses") is True
    assert is_pulsemap_check("InIceDSTPulses") is True
    assert is_pulsemap_check("RTTWOfflinePulses") is True
    assert is_pulsemap_check("truth") is False
    assert is_pulsemap_check("retro") is False


@pytest.mark.order(1)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataconverter(
    backend: str, test_data_dir: str = TEST_DATA_DIR
) -> None:
    """Test the implementation of `DataConverter` for `backend`."""
    # Constructor DataConverter instance
    opt = dict(
        extractors=[
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86("SRTInIcePulses"),
        ],
        outdir=OUTPUT_DATA_DIR,
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

    # Check output
    path = get_file_path(backend)
    assert os.path.exists(path), path


@pytest.mark.order(3)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataset(backend: str) -> None:
    """Test the implementation of `Dataset` for `backend`."""
    path = get_file_path(backend)
    assert os.path.exists(path)

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
        assert len(event.features) == len(opt["features"])


@pytest.mark.order(4)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataset_query_table(backend: str) -> None:
    """Test the implementation of `Dataset._query_table` for `backend`."""
    path = get_file_path(backend)
    assert os.path.exists(path)

    # Constructor DataConverter instance
    pulsemap = "SRTInIcePulses"
    opt = dict(
        path=path,
        pulsemaps=pulsemap,
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
    )

    if backend == "sqlite":
        dataset = SQLiteDataset(**opt)  # type: ignore[arg-type]
    elif backend == "parquet":
        dataset = ParquetDataset(**opt)  # type: ignore[arg-type]
    else:
        assert False, "Shouldn't reach here"

    # Compare to expectations
    nb_events_to_test = 5
    results_all = dataset._query_table(
        pulsemap,
        columns=["event_no", opt["features"][0]],
    )
    for ix_test in range(nb_events_to_test):

        results_single = dataset._query_table(
            pulsemap,
            columns=["event_no", opt["features"][0]],
            sequential_index=ix_test,
        )
        event_nos = list(set([res[0] for res in results_single]))
        assert len(event_nos) == 1
        event_no: int = event_nos[0]
        results_all_subset = [res for res in results_all if res[0] == event_no]
        assert results_all_subset == results_single


if __name__ == "__main__":
    test_dataset_query_table("parquet")
