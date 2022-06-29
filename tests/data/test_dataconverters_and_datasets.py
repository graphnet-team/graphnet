"""Unit tests for SQLiteDataConverter."""
import os

import pytest

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
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
from graphnet.utilities.imports import requires_icecube


# Global variable(s)
TEST_DATA_DIR = os.path.abspath("./test_data/")
FILE_NAME = "oscNext_genie_level7_v03.01_pass2.160000.000001"
GCD_FILE = (
    "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
)


# Utility method(s)
def get_file_path(backend: str, test_data_dir: str) -> str:
    """Return the path to the output file for `backend`."""
    suffix = {
        "sqlite": ".db",
        "parquet": ".parquet",
    }[backend]

    path = os.path.join(test_data_dir, FILE_NAME + suffix)
    return path


# Unit test(s)
def test_is_pulsemap_check():
    assert is_pulsemap_check("SplitInIcePulses") is True
    assert is_pulsemap_check("SRTInIcePulses") is True
    assert is_pulsemap_check("InIceDSTPulses") is True
    assert is_pulsemap_check("RTTWOfflinePulses") is True
    assert is_pulsemap_check("truth") is False
    assert is_pulsemap_check("retro") is False


# @requires_icecube
@pytest.mark.order(1)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataconverter(backend: str, test_data_dir: str = TEST_DATA_DIR):
    """Test the implementation of `DataConverter` for `backend`."""
    # Constructor DataConverter instance
    opt = dict(
        extractors=[
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86("SRTInIcePulses"),
        ],
        outdir=test_data_dir,
        gcd_rescue=os.path.join(
            test_data_dir,
            GCD_FILE,
        ),
        workers=1,
    )

    if backend == "sqlite":
        converter = SQLiteDataConverter(**opt)
    elif backend == "parquet":
        converter = ParquetDataConverter(**opt)
    else:
        assert False, "Shouldn't reach here"

    # Perform conversion from I3 to `backend`
    converter(test_data_dir)

    # Check output
    path = get_file_path(backend, test_data_dir)
    assert os.path.exists(path)


# @requires_icecube
@pytest.mark.order(2)
@pytest.mark.parametrize("backend", ["sqlite", "parquet"])
def test_dataset(backend: str, test_data_dir: str = TEST_DATA_DIR):
    """Test the implementation of `Dataset` for `backend`."""
    path = get_file_path(backend, test_data_dir)
    assert os.path.exists(path)

    # Constructor DataConverter instance
    opt = dict(
        path=path,
        pulsemaps="SRTInIcePulses",
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
    )

    if backend == "sqlite":
        dataset = SQLiteDataset(**opt)
    elif backend == "parquet":
        dataset = ParquetDataset(**opt)
    else:
        assert False, "Shouldn't reach here"

    # Compare to expectations
    expected_number_of_events = 5552
    test_indices = list(range(10)) + list(
        range(expected_number_of_events - 10, expected_number_of_events)
    )
    expected_numbers_of_pulses = [
        11,
        13,
        9,
        13,
        8,
        8,
        9,
        9,
        9,
        8,
        147,
        21,
        238,
        469,
        181,
        22,
        47,
        121,
        18,
        246,
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
