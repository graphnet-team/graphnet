"""Example of converting Parquet files to SQLite database."""

from graphnet.constants import TEST_DATA_DIR, TEST_PARQUET_DATA
from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter
from graphnet.utilities.argparse import ArgumentParser


def main() -> None:
    """Run example."""
    # Path to parquet file or directory containing parquet files
    parquet_path = TEST_PARQUET_DATA

    # Path to where you want the database to be stored
    outdir = f"{TEST_DATA_DIR}/output/convert_parquet_to_sqlite/"

    # Name of the database.
    # Will be saved in outdir/database_name/data/database_name.db
    database_name = TEST_PARQUET_DATA.split("/")[-1].split(".")[0]

    converter = ParquetToSQLiteConverter(
        mc_truth_table="truth",
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

    args = parser.parse_args()

    main()
