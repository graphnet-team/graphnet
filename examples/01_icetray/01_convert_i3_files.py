"""Example of converting I3-files to SQLite and Parquet."""

import os

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors import (
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
    I3GenericExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

from _common_icetray import ERROR_MESSAGE_MISSING_ICETRAY

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

    converter: DataConverter = CONVERTER_CLASS[backend](
        [
            I3GenericExtractor(
                keys=[
                    "SRTInIcePulses",
                    "I3MCTree",
                ]
            ),
            I3TruthExtractor(),
        ],
        outdir,
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
    workers = 1

    converter: DataConverter = CONVERTER_CLASS[backend](
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCubeUpgrade("I3RecoPulseSeriesMap_mDOM"),
            I3FeatureExtractorIceCubeUpgrade("I3RecoPulseSeriesMap_DEgg"),
        ],
        outdir,
        workers=workers,
        # nb_files_to_batch=10,
        # sequential_batch_pattern="temp_{:03d}",
        # input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        icetray_verbose=1,
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

        args = parser.parse_args()

        # Run example script
        if args.detector == "icecube-86":
            main_icecube86(args.backend)
        else:
            main_icecube_upgrade(args.backend)
