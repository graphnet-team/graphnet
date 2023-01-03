"""Example of converting Parquet files to SQLite database."""

from graphnet.constants import TEST_DATA_DIR, TEST_PARQUET_DATA
from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter
from graphnet.utilities.argparse import ArgumentParser


def main(parquet_path: str, mc_truth_table: str) -> None:
    """Run example."""
    # Path to where you want the database to be stored
    outdir = f"{TEST_DATA_DIR}/output/convert_parquet_to_sqlite/"

    # Name of the database.
    # Will be saved in {outdir}/{database_name}/data/{database_name}.db
    database_name = parquet_path.split("/")[-1].split(".")[0]

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
