"""Example of converting I3-files to SQLite, Parquet, and LMDB.

When using the LMDB backend, the ``--precompute-representation`` flag can be
used to pre-compute a DataRepresentation and store it alongside the raw
data. Pre-computed representations can later be loaded directly,
avoiding the cost of real-time DataRepresentation construction during training.
"""

from glob import glob
from typing import Any, Dict
from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCubeUpgrade,
    I3FeatureExtractorIceCube86,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter
from graphnet.data.pre_configured.dataconverters import I3ToLMDBConverter
from graphnet.models.detector.icecube import IceCube86, IceCubeUpgrade
from graphnet.models.graphs import KNNGraph
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
    "lmdb": I3ToLMDBConverter,
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}


def main_icecube86(
    backend: str,
    precompute_representation: bool = False,
    num_workers: int = 1,
) -> None:
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    assert backend in CONVERTER_CLASS

    inputs = [f"{TEST_DATA_DIR}/i3/oscNext_genie_level7_v02"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/ic86"
    gcd_rescue = glob(
        f"{TEST_DATA_DIR}/i3/oscNext_genie_level7_v02/*GeoCalib*"
    )[0]

    extractors = [
        I3FeatureExtractorIceCube86("SRTInIcePulses"),
        I3TruthExtractor(),
    ]

    if backend == "lmdb":
        lmdb_kwargs: Dict[str, Any] = {}
        if precompute_representation:
            # Could be any DataRepresentation, not just KNNGraph
            data_representation = KNNGraph(
                detector=IceCube86(),
                nb_nearest_neighbours=8,
                input_feature_names=FEATURES.ICECUBE86,
            )
            lmdb_kwargs.update(
                data_representation=data_representation,
                pulsemap_extractor_name="SRTInIcePulses",
                truth_extractor_name="truth",
                truth_label_names=TRUTH.ICECUBE86,
            )
        converter: DataConverter = I3ToLMDBConverter(
            extractors=extractors,
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            num_workers=num_workers,
            **lmdb_kwargs,
        )
    else:
        converter = CONVERTER_CLASS[backend](
            extractors=extractors,
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            workers=num_workers,
        )

    converter(inputs)
    if backend in ["sqlite", "lmdb"]:
        converter.merge_files()


def main_icecube_upgrade(
    backend: str,
    precompute_representation: bool = False,
    num_workers: int = 1,
) -> None:
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    assert backend in CONVERTER_CLASS

    inputs = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/upgrade"
    gcd_rescue = glob(
        "{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/*GeoCalib*"
    )[0]

    pulsemap = "I3RecoPulseSeriesMap_mDOM"
    extractors = [
        I3TruthExtractor(),
        I3RetroExtractor(),
        I3FeatureExtractorIceCubeUpgrade(pulsemap),
        I3FeatureExtractorIceCubeUpgrade("I3RecoPulseSeriesMap_DEgg"),
    ]

    if backend == "lmdb":
        lmdb_kwargs: Dict[str, Any] = {}
        if precompute_representation:
            data_representation = KNNGraph(
                detector=IceCubeUpgrade(),
                nb_nearest_neighbours=8,
                input_feature_names=FEATURES.UPGRADE,
            )
            lmdb_kwargs.update(
                data_representation=data_representation,
                pulsemap_extractor_name=pulsemap,
                truth_extractor_name="truth",
                truth_label_names=TRUTH.UPGRADE,
            )
        converter: DataConverter = I3ToLMDBConverter(
            extractors=extractors,
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            num_workers=num_workers,
            **lmdb_kwargs,
        )
    else:
        converter = CONVERTER_CLASS[backend](
            extractors=extractors,
            outdir=outdir,
            gcd_rescue=gcd_rescue,
            workers=num_workers,
        )

    converter(inputs)
    if backend in ["sqlite", "lmdb"]:
        converter.merge_files()


if __name__ == "__main__":

    if not has_icecube_package():
        Logger(log_folder=None).error(ERROR_MESSAGE_MISSING_ICETRAY)
    else:
        parser = ArgumentParser(
            description="""
Convert I3 files to an intermediate format.
"""
        )

        parser.add_argument(
            "backend",
            nargs="?",
            choices=["lmdb", "sqlite", "parquet"],
            default="lmdb",
            help="Backend format to convert to (default: %(default)s)",
        )
        parser.add_argument(
            "detector", choices=["icecube-86", "icecube-upgrade"]
        )
        parser.add_argument(
            "--precompute-representation",
            action="store_true",
            default=False,
            help="Pre-compute a KNN graph representation and store it in "
            "the LMDB database. Only supported with the lmdb backend.",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Number of worker processes for parallel conversion "
            "(default: %(default)s).",
        )

        args, unknown = parser.parse_known_args()

        if args.precompute_representation and args.backend != "lmdb":
            Logger(log_folder=None).warning(
                "--precompute-representation is only supported with the lmdb "
                "backend. Ignoring."
            )
            args.precompute_representation = False

        if args.detector == "icecube-86":
            main_icecube86(
                args.backend,
                args.precompute_representation,
                args.workers,
            )
        else:
            main_icecube_upgrade(
                args.backend,
                args.precompute_representation,
                args.workers,
            )
