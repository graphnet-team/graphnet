"""Module containing `GraphNeTFileSaveMethod`(s).

These modules are used to save the interim data format from `DataConverter` to
a deep-learning friendly file format.
"""

import os
from tqdm import tqdm
from typing import List, Dict, Optional

from graphnet.data.utilities import (
    create_table_and_save_to_sql,
    get_primary_keys,
    query_database,
)
import pandas as pd
from .graphnet_writer import GraphNeTWriter


class SQLiteWriter(GraphNeTWriter):
    """A method for saving GraphNeT's interim dataformat to SQLite."""

    def __init__(
        self,
        merged_database_name: str = "merged.db",
        max_table_size: Optional[int] = None,
    ) -> None:
        """Initialize `SQLiteWriter`.

        Args:
            merged_database_name: name of the database, not path, that files
                                  will be merged into. Defaults to "merged.db".
            max_table_size: The maximum number of rows in any given table.
                If given, the merging proceedure splits the databases into
                partitions each with a maximum table size of max_table_size.
                Note that the size is approximate. This feature is useful if
                you have many events, as tables exceeding
                400 million rows tend to be noticably slower to query.
                Defaults to None (All events are put into a single database).
        """
        # Member Variables
        self._file_extension = ".db"
        self._merge_dataframes = True
        self._max_table_size = max_table_size
        self._database_name = merged_database_name

        # Add file extension to database name if forgotten
        if not self._database_name.endswith(self._file_extension):
            self._database_name = self._database_name + self._file_extension

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def _save_file(
        self,
        data: Dict[str, pd.DataFrame],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to SQLite database."""
        # Check(s)
        if os.path.exists(output_file_path):
            self.warning(
                f"Output file {output_file_path} already exists. Appending."
            )

        # Concatenate data
        if len(data) == 0:
            self.warning(
                "No data was extracted from the processed I3 file(s). "
                f"No data saved to {output_file_path}"
            )
            return

        saved_any = False
        # Save each dataframe to SQLite database
        self.debug(f"Saving to {output_file_path}")
        for table, df in data.items():
            if len(df) > 0:
                create_table_and_save_to_sql(
                    df,
                    table,
                    output_file_path,
                    default_type="FLOAT",
                    integer_primary_key=len(df) <= n_events,
                )
                saved_any = True

        if saved_any:
            self.debug("- Done saving")
        else:
            self.warning(f"No data saved to {output_file_path}")

    def merge_files(
        self,
        files: List[str],
        output_dir: str,
        primary_key_rescue: str = "event_no",
    ) -> None:
        """SQLite-specific method for merging output files/databases.

        Args:
            files: paths to SQLite databases that needs to be merged.
            output_dir: path to store the merged database(s) in.
            database_name: name, not path, of database. E.g. "my_database".
            max_table_size: The maximum number of rows in any given table.
                If given, the merging proceedure splits the databases into
                partitions each with a maximum table size of max_table_size.
                Note that the size is approximate. This feature is useful if
                you have many events, as tables exceeding
                400 million rows tend to be noticably slower to query.
                Defaults to None (All events are put into a single database.)
            primary_key_rescue: The name of the columns on which the primary
                key is constructed. This will only be used if it is not
                possible to infer the primary key name.
        """
        # Warnings
        if self._max_table_size:
            self.warning(
                f"Merging got max_table_size of {self._max_table_size}."
                " Will attempt to create databases with a maximum row count of"
                " this size."
            )

        # Set variables
        self._partition_count = 1
        self._primary_key_rescue = primary_key_rescue

        # Construct full database path
        database_path = os.path.join(output_dir, self._database_name)
        # Start merging if files are given
        if len(files) > 0:
            os.makedirs(output_dir, exist_ok=True)
            self.info(f"Merging {len(files)} database files")
            self._merge_databases(files=files, database_path=database_path)
        else:
            self.warning("No database files given! Exiting.")

    def _merge_databases(
        self,
        files: List[str],
        database_path: str,
    ) -> None:
        """Merge the temporary databases.

        Args:
            files: List of files to be merged.
            database_path: Path to a database, can be an empty path, where the
            databases listed in `files` will be merged into. If no database
            exists at the given path, one will be created.
        """
        if os.path.exists(database_path):
            self.warning(
                "Target path for merged database",
                f"{database_path}, already exists.",
            )

        if self._max_table_size is not None:
            database_path = self._adjust_output_path(database_path)
            self._row_counts: Dict[str, int] = {}
            self._largest_table = 0

        # Merge temporary databases into newly created one
        for file_count, input_file in tqdm(enumerate(files), colour="green"):
            # Extract table names and index column name in database
            try:
                tables, primary_key = get_primary_keys(database=input_file)
                if primary_key is None:
                    primary_key = self._primary_key_rescue
            except AssertionError as e:
                if "No tables found in database." in str(e):
                    self.warning(f"Database {input_file} is empty. Skipping.")
                    continue
                else:
                    raise e

            for table_name in tables.keys():
                # Extract all data in the table from the given database
                df = query_database(
                    database=input_file, query=f"SELECT * FROM {table_name}"
                )

                # Infer whether the table was previously indexed with
                # A primary key or not. len(tables[table]) = 0 if not.
                integer_primary_key = (
                    True if tables[table_name] is not None else False
                )

                # Submit to new database
                create_table_and_save_to_sql(
                    df=df,
                    table_name=table_name,
                    database_path=database_path,
                    index_column=primary_key,
                    integer_primary_key=integer_primary_key,
                    default_type="FLOAT",
                )

                # Update row counts if needed
                if self._max_table_size is not None:
                    self._update_row_counts(df=df, table_name=table_name)

            if (self._max_table_size is not None) & (file_count < len(files)):
                assert self._max_table_size is not None  # mypy...
                if self._largest_table >= self._max_table_size:
                    # Increment partition, reset counts, adjust output path
                    self._partition_count += 1
                    self._row_counts = {}
                    self._largest_table = 0
                    database_path = self._adjust_output_path(database_path)
                    self.info(
                        "Maximum row count reached."
                        f" Creating new partition at {database_path}"
                    )

    # Internal methods

    def _adjust_output_path(self, output_file: str) -> str:
        """Adjust the file path to reflect that it is a partition."""
        path_without_extension, extension = os.path.splitext(output_file)
        if "_part_" in path_without_extension:
            # if true, this is already a partition.
            database_name = path_without_extension.split("_part_")[:-1][0]
        else:
            database_name = path_without_extension
        # split into multiple lines to avoid one long
        database_name = database_name + f"_part_{self._partition_count}"
        database_name = database_name + extension
        return database_name

    def _update_row_counts(self, df: pd.DataFrame, table_name: str) -> None:
        if table_name in self._row_counts.keys():
            self._row_counts[table_name] += len(df)
        else:
            self._row_counts[table_name] = len(df)

        self._largest_table = max(self._row_counts.values())
        return
