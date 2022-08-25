"""Minimum working example (MWE) to use ParquetDataConverter."""

import logging
import os

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
    I3GenericExtractor,
)
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter

logger = get_logger(level=logging.INFO)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_icecube86(backend: str):
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    paths = [
        # "/groups/icecube/asogaard/data/i3/i3_to_sqlite_workshop_test/level7_v02.00"
        "./test_data/"
    ]

    # gcd_rescue = "resources/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
    gcd_rescue = os.path.join(
        paths[0],
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
    )
    outdir = "/groups/icecube/asogaard/temp/parquet_test_ic86"

    converter = CONVERTER_CLASS[backend](
        [
            I3GenericExtractor(),
        ],
        outdir,
        gcd_rescue,
    )
    converter(paths)


def main_icecube_upgrade(backend: str):
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    # basedir = "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/"
    basedir = "test_data_upgrade_2"
    # paths = [os.path.join(basedir, "step4")]
    paths = [basedir]
    gcd_rescue = os.path.join(
        # basedir, "gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
        basedir,
        "GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2",
    )
    outdir = "/groups/icecube/asogaard/temp/parquet_test_upgrade"
    workers = 1

    converter = CONVERTER_CLASS[backend](
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_mDOM"
            ),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_DEgg"
            ),
        ],
        outdir,
        gcd_rescue,
        workers=workers,
        # nb_files_to_batch=10,
        # sequential_batch_pattern="temp_{:03d}",
        # input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        icetray_verbose=1,
    )
    converter(paths)


if __name__ == "__main__":
    backend = "parquet"
    # backend = "sqlite"
    main_icecube86(backend)
    # main_icecube_upgrade(backend)
