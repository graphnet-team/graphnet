"""Example of converting Parquet files to SQLite database."""

from graphnet.data.utilities.parquet_to_sqlite import ParquetToSQLiteConverter


def main() -> None:
    """Run example."""
    # Path to parquet file or directory containing parquet files
    parquet_path = "/my_file.parquet"

    # Path to where you want the database to be stored
    outdir = "/home/my_databases/"

    # Name of the database.
    # Will be saved in outdir/database_name/data/database_name.db
    database_name = "my_database_from_parquet"

    converter = ParquetToSQLiteConverter(
        mc_truth_table="mc_truth", parquet_path=parquet_path
    )
    converter.run(outdir=outdir, database_name=database_name)


if __name__ == "__main__":
    main()
