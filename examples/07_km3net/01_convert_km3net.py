"""Code to run the extraction of km3net data."""

import os
import warnings

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.readers import KM3NeTReader
from graphnet.data.writers import ParquetWriter, SQLiteWriter
from graphnet.data import DataConverter
from graphnet.data.extractors.km3net import (
    KM3NeTTruthExtractor,
    KM3NeTFullPulseExtractor,
    KM3NeTTriggPulseExtractor,
    KM3NeTHNLTruthExtractor,
    KM3NeTRegularRecoExtractor,
    KM3NeTHNLRecoExtractor,
)

from graphnet.utilities.argparse import ArgumentParser


def main(backend: str, triggered: str, HNL: str, OUTPUT_DIR: str) -> None:
    """Convert ROOT files from KM3NeT to `backend` format."""
    warnings.simplefilter(action="ignore", category=FutureWarning)

    input_dir = [f"{TEST_DATA_DIR}/km3net"]
    if OUTPUT_DIR != "None":
        outdir = f"{OUTPUT_DIR}/{backend}"
    else:
        outdir = f"{EXAMPLE_OUTPUT_DIR}/{backend}"
    os.makedirs(outdir, exist_ok=True)
    print(60*'*')
    print(f"Saving to {outdir}")
    print(60*'*')
    if backend == "parquet":
        save_method = ParquetWriter(truth_table="truth")
    elif backend == "sqlite":
        save_method = SQLiteWriter()  # type: ignore
    else:
        raise ValueError("Invalid backend choice")

    if HNL == "km3net-vars":
        truth_extractor = KM3NeTTruthExtractor(name="truth")
        reco_extractor = KM3NeTRegularRecoExtractor(name="reco")
    elif HNL == "hnl-vars":
        truth_extractor = KM3NeTHNLTruthExtractor(name="truth")  # type: ignore
        reco_extractor = KM3NeTHNLRecoExtractor(name="reco")  # type: ignore
    else:
        raise ValueError("Invalid HNL choice")

    if triggered == "Triggered":
        pulse_extractor = KM3NeTTriggPulseExtractor(name="pulse_map")
    elif triggered == "Snapshot":
        pulse_extractor = KM3NeTFullPulseExtractor(
            name="pulse_map"
        )  # type: ignore
    else:
        raise ValueError("Invalid triggered choice")

    converter = DataConverter(
        file_reader=KM3NeTReader(),
        save_method=save_method,
        extractors=[truth_extractor, pulse_extractor, reco_extractor],
        outdir=outdir,
        num_workers=1,
    )

    converter(input_dir=input_dir)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Convert root files from KM3NeT to an sqlite or parquet.
            """
    )

    parser.add_argument(
        "backend",
        choices=["sqlite", "parquet"],
        help="Choose the backend format",
    )
    parser.add_argument(
        "triggered",
        choices=["Triggered", "Snapshot"],
        help="Choose between triggered or snapshot pulse maps",
    )
    parser.add_argument(
        "HNL",
        choices=["km3net-vars", "hnl-vars"],
        help="Km3net truth or adding Heavy Neutral Lepton info",
    )
    parser.add_argument(
        "OUTPUT_DIR",
        default="None",
        help="Output directory (optional)",
    )

    args, unknown = parser.parse_known_args()

    # Run example script
    main(args.backend, args.triggered, args.HNL, args.OUTPUT_DIR)
