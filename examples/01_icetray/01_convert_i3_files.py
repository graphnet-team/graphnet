"""Example of converting I3-files to SQLite and Parquet."""

import os
from glob import glob

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCubeUpgrade,
    I3FeatureExtractorIceCube86,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

ERROR_MESSAGE_MISSING_ICETRAY = (
    "This example requires IceTray to be installed, which doesn't seem to be "
    "the case. Please install IceTray; run this example in the GraphNeT "
    "Docker container which comes with IceTray installed; or run an example "
    "script in one of the other folders:"
    "\n * examples/02_data/"
    "\n * examples/03_weights/"
    "\n * examples/04_training/"
    "\n * examples/05_pisa/"
    "\nExiting."
)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_icecube86(backend: str) -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = [f"{TEST_DATA_DIR}/i3/oscNext_genie_level7_v02"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/ic86"
    gcd_rescue = glob(
        "{TEST_DATA_DIR}/i3/oscNext_genie_level7_v02/*GeoCalib*"
    )[0]

    converter = CONVERTER_CLASS[backend](
        extractors=[
            I3FeatureExtractorIceCube86("SRTInIcePulses"),
            I3TruthExtractor(),
        ],
        outdir=outdir,
        gcd_rescue=gcd_rescue,
        workers=1,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


def main_icecube_upgrade(backend: str) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/upgrade"
    gcd_rescue = glob(
        "{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/*GeoCalib*"
    )[0]
    workers = 1

    converter: DataConverter = CONVERTER_CLASS[backend](
        extractors=[
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCubeUpgrade("I3RecoPulseSeriesMap_mDOM"),
            I3FeatureExtractorIceCubeUpgrade("I3RecoPulseSeriesMap_DEgg"),
        ],
        outdir=outdir,
        workers=workers,
        gcd_rescue=gcd_rescue,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        # Parse command-line arguments
        parser = ArgumentParser(
            description="""
Convert I3 files to an intermediate format.
"""
        )

        parser.add_argument("backend", choices=["sqlite", "parquet"])
        parser.add_argument(
            "detector", choices=["icecube-86", "icecube-upgrade"]
        )

        args, unknown = parser.parse_known_args()

        # Run example script
        if args.detector == "icecube-86":
            main_icecube86(args.backend)
        else:
            main_icecube_upgrade(args.backend)
