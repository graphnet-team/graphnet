"""Unit tests for SQLiteDataConverter."""
import os

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3RetroExtractor,
)
from graphnet.data.sqlite.sqlite_dataconverter import (
    SQLiteDataConverter,
    is_pulsemap_check,
)
from graphnet.data.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.utilities.imports import requires_icecube

TEST_DATA_DIR = os.path.abspath("./test_data/")


def test_is_pulsemap_check():
    assert is_pulsemap_check("SplitInIcePulses") is True
    assert is_pulsemap_check("SRTInIcePulses") is True
    assert is_pulsemap_check("InIceDSTPulses") is True
    assert is_pulsemap_check("RTTWOfflinePulses") is True
    assert is_pulsemap_check("truth") is False
    assert is_pulsemap_check("retro") is False


@requires_icecube
def test_sqlite_database(test_data_dir: str = TEST_DATA_DIR):
    database = os.path.join(
        test_data_dir, "oscNext_genie_level7_v03.01_pass2.160000.000001.db"
    )
    pulsemap = "SRTInIcePulses"
    features = FEATURES.DEEPCORE
    truth = TRUTH.DEEPCORE

    dataset = SQLiteDataset(
        database,
        pulsemap,
        features,
        truth,
    )

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
        assert len(event.features) == len(features)


@requires_icecube
def convert_i3_to_sqlite(test_data_dir: str = TEST_DATA_DIR):
    """..."""
    paths = test_data_dir
    pulsemap = "SRTInIcePulses"
    gcd_rescue = os.path.join(
        test_data_dir,
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
    )
    outdir = test_data_dir
    workers = 1

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(pulsemap),
        ],
        outdir,
        gcd_rescue,
        workers=workers,
    )
    converter(paths)
    return


if __name__ == "__main__":
    convert_i3_to_sqlite()
