"""Unit tests for I3GenericExtractor class."""

import os

import numpy as np

import graphnet.constants
from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3TruthExtractor,
    I3GenericExtractor,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
    from icecube import dataio  # pyright: reportMissingImports=false

# Global variable(s)
TEST_DATA_DIR = os.path.join(
    graphnet.constants.TEST_DATA_DIR, "i3", "oscNext_genie_level7_v02"
)
FILE_NAME = "oscNext_genie_level7_v02_first_5_frames"
GCD_FILE = (
    "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
)


# Unit test(s)
def test_i3genericextractor(test_data_dir: str = TEST_DATA_DIR) -> None:
    """Test the implementation of `I3GenericExtractor`."""
    # Constants(s)
    mc_tree = "I3MCTree"
    pulse_series = "SRTInIcePulses"

    # Constructor I3Extractor instance(s)
    generic_extractor = I3GenericExtractor(keys=[mc_tree, pulse_series])
    truth_extractor = I3TruthExtractor()
    feature_extractor = I3FeatureExtractorIceCube86(pulse_series)

    i3_file = os.path.join(test_data_dir, FILE_NAME) + ".i3.gz"
    gcd_file = os.path.join(test_data_dir, GCD_FILE)

    generic_extractor.set_files(i3_file, gcd_file)
    truth_extractor.set_files(i3_file, gcd_file)
    feature_extractor.set_files(i3_file, gcd_file)

    i3_file_io = dataio.I3File(i3_file, "r")
    ix_test = 5
    while i3_file_io.more():
        try:
            frame = i3_file_io.pop_physics()
        except:  # noqa: E722
            continue

        generic_data = generic_extractor(frame)
        truth_data = truth_extractor(frame)
        feature_data = feature_extractor(frame)

        if ix_test == 5:
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
            ("rde", "relative_dom_eff"),
        ]

        for reco_key, generic_key in key_pairs:
            assert np.allclose(
                feature_data[reco_key], generic_data[pulse_series][generic_key]
            )

        ix_test -= 1
        if ix_test == 0:
            break
