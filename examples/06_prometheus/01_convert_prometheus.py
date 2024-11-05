"""Example of converting files from Prometheus to SQLite and Parquet."""

import os

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR
from graphnet.data.extractors.prometheus import (
    PrometheusTruthExtractor,
    PrometheusFeatureExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import PrometheusReader
from graphnet.data.writers import ParquetWriter, SQLiteWriter
from graphnet.utilities.argparse import ArgumentParser


def main(backend: str) -> None:
    """Convert parquet files from Prometheus to `backend` format."""
    # Fixed inputs
    input_dir = [f"{TEST_DATA_DIR}/prometheus"]
    outdir = f"{EXAMPLE_OUTPUT_DIR}/prometheus/{backend}"
    os.makedirs(outdir, exist_ok=True)
    num_workers = 1

    if backend == "parquet":
        save_method = ParquetWriter(truth_table="mc_truth")
    elif backend == "sqlite":
        save_method = SQLiteWriter()  # type: ignore

    converter = DataConverter(
        file_reader=PrometheusReader(),
        save_method=save_method,
        extractors=[PrometheusTruthExtractor(), PrometheusFeatureExtractor()],
        outdir=outdir,
        num_workers=num_workers,
    )

    converter(input_dir=input_dir)

    converter.merge_files()


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Convert parquet files from Prometheus to an intermediate format.
            """
    )

    parser.add_argument("backend", choices=["sqlite", "parquet"])

    args, unknown = parser.parse_known_args()

    # Run example script
    main(args.backend)
