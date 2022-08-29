"""Minimum working example (MWE) to use ParquetDataConverter."""

import logging
import os

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.parquet.parquet_dataconverter import ParquetDataConverter

logger = get_logger(level=logging.INFO)


def main_icecube86():
    """Main script function."""
    paths = [
        "/groups/icecube/asogaard/data/i3/i3_to_sqlite_workshop_test/level7_v02.00"
    ]
    pulsemap = "SRTInIcePulses"
    gcd_rescue = "resources/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
    outdir = "/groups/icecube/asogaard/temp/parquet_test_ic86"

    converter = ParquetDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(pulsemap),
        ],
        outdir,
        gcd_rescue,
    )
    converter(paths)


def main_icecube_upgrade():
    """Main script function."""
    basedir = "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/"
    paths = [os.path.join(basedir, "step4")]
    gcd_rescue = os.path.join(
        basedir, "gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
    )
    outdir = "/groups/icecube/asogaard/temp/parquet_test_upgrade"
    workers = 10

    converter = ParquetDataConverter(
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
        nb_files_to_batch=1000,
        # sequential_batch_pattern="temp_{:03d}",
        input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        verbose=1,
    )
    converter(paths)


if __name__ == "__main__":
    # main_icecube86()
    main_icecube_upgrade()
