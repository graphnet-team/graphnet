"""Example of converting Parquet files to SQLite database."""

import os.path

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_PARQUET_DATA
from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger


def main(parquet_path: str, mc_truth_table: str) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Path to where you want the database to be stored
    outdir = f"{EXAMPLE_OUTPUT_DIR}/convert_parquet_to_sqlite/"

    # Name of the database.
    # Will be saved in {outdir}/{database_name}/data/{database_name}.db
    database_name = parquet_path.split("/")[-1].split(".")[0]
    output_path = f"{outdir}/{database_name}/data/{database_name}.db"

    if os.path.exists(output_path):
        logger.error(
            f"Output database {output_path} already exists. The conversion "
            "will likely fail. If so, please remove the output database "
            "before running this script again."
        )

    converter = ParquetToSQLiteConverter(
        mc_truth_table=mc_truth_table,
        parquet_path=parquet_path,
    )
    converter.run(outdir=outdir, database_name=database_name)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Convert Parquet files to SQLite database.
"""
    )

    parser.add_argument(
        "--parquet-path",
        action="store",
        help="Path to Parquet file (default: %(default)s)",
        default=TEST_PARQUET_DATA,
    )
    parser.add_argument(
        "--mc-truth-table",
        action="store",
        help="Name of MC truth table in Parquet file (default: %(default)s)",
        default="truth",
    )

    args = parser.parse_args()

    main(args.parquet_path, args.mc_truth_table)
