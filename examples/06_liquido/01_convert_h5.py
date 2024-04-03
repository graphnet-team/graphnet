"""Example of converting I3-files to SQLite and Parquet."""

import os
from glob import glob

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.liquido import H5Extractor
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import LiquidOReader
from graphnet.data.writers import ParquetWriter, SQLiteWriter
from graphnet.utilities.argparse import ArgumentParser


def main(backend: str) -> None:
    """Convert h5 files from LiquidO to intermediate `backend` format."""
    # Fixed inputs
    input_dir = [f"{TEST_DATA_DIR}/liquid-o"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/liquid-o/{backend}"
    num_workers = 1

    if backend == "parquet":
        save_method = ParquetWriter(truth_table="TruthData")
    elif backend == "sqlite":
        save_method = SQLiteWriter()

    truth_extractor = H5Extractor(
        "TruthData", ["event_no", "x", "y", "z", "ze", "az", "t", "E", "pid"]
    )
    hit_extractor = H5Extractor(
        "HitData", ["event_no", "sipmID", "x", "y", "z", "t", "var"]
    )
    converter = DataConverter(
        file_reader=LiquidOReader(),
        save_method=save_method,
        extractors=[truth_extractor, hit_extractor],
        outdir=outdir,
        num_workers=num_workers,
    )

    converter(input_dir=input_dir)

    converter.merge_files()


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Convert h5 files from LiquidO to an intermediate format.
            """
    )

    parser.add_argument("backend", choices=["sqlite", "parquet"])

    args, unknown = parser.parse_known_args()

    # Run example script
    main(args.backend)
