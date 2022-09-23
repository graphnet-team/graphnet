"""Unit tests for SQLiteDataConverter."""
import os

import numpy as np
import pytest

from graphnet.data.constants import FEATURES, TRUTH
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
    assert os.path.exists(path), path


def test_i3genericextractor(test_data_dir: str = TEST_DATA_DIR):
    """Test the implementation of `I3GenericExtractor`."""

    # Constants(s)
    mc_tree = "I3MCTree"
    pulse_series = "SRTInIcePulses"

    # Constructor I3Extractor instance(s)
    generic_extractor = I3GenericExtractor(keys=[mc_tree, pulse_series])
    truth_extractor = I3TruthExtractor()
    feature_extractor = I3FeatureExtractorIceCube86(pulse_series)

    i3_file = os.path.join(test_data_dir, FILE_NAME) + ".i3.zst"
    gcd_file = os.path.join(test_data_dir, GCD_FILE)

    generic_extractor.set_files(i3_file, gcd_file)
    truth_extractor.set_files(i3_file, gcd_file)
    feature_extractor.set_files(i3_file, gcd_file)

    i3_file_io = dataio.I3File(i3_file, "r")
    ix_test = 10
    while i3_file_io.more():
        try:
            frame = i3_file_io.pop_physics()
        except:  # noqa: E722
            continue

        generic_data = generic_extractor(frame)
        truth_data = truth_extractor(frame)
        feature_data = feature_extractor(frame)

        if ix_test == 10:
            print(list(generic_data[pulse_series].keys()))
            print(list(truth_data.keys()))
            print(list(feature_data.keys()))

        # Truth vs. generic
        key_pairs = [
            ("energy", "energy"),
            ("zenith", "dir__zenith"),
            ("azimuth", "dir__azimuth"),
            ("pid", "pdg_encoding"),
        ]

        for truth_key, generic_key in key_pairs:
            assert (
                truth_data[truth_key]
                == generic_data[f"{mc_tree}__primaries"][generic_key][0]
            )

        # Reco vs. generic
        key_pairs = [
            ("charge", "charge"),
            ("dom_time", "time"),
            ("dom_x", "position__x"),
            ("dom_y", "position__y"),
            ("dom_z", "position__z"),
            ("width", "width"),
            ("pmt_area", "area"),
            # ("rde", "relative_dom_efficiency"),  <-- Missing
        ]

        for reco_key, generic_key in key_pairs:
            assert np.allclose(
                feature_data[reco_key], generic_data[pulse_series][generic_key]
            )

        ix_test -= 1
        if ix_test == 0:
            break


@pytest.mark.order(3)
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


if __name__ == "__main__":
    # test_dataconverter("sqlite")
    test_i3genericextractor()
