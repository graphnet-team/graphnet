"""DataConverter for the SQLite backend."""

from collections import OrderedDict
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import sqlalchemy
import sqlite3
from tqdm import tqdm

from graphnet.data.dataconverter import DataConverter  # type: ignore[attr-defined]
from graphnet.data.sqlite.sqlite_utilities import (
    create_table,
    create_table_and_save_to_sql,
)


class SQLiteDataConverter(DataConverter):
    """Class for converting I3-file(s) to SQLite format."""

    # Class variables
    file_suffix = "db"

    # Abstract method implementation(s)
    def save_data(self, data: List[OrderedDict], output_file: str) -> None:
        """Save data to SQLite database."""
        # Check(s)
        if os.path.exists(output_file):
            self.warning(
                f"Output file {output_file} already exists. Appending."
            )

        # Concatenate data
        if len(data) == 0:
            self.warning(
                "No data was extracted from the processed I3 file(s). "
                f"No data saved to {output_file}"
            )
            return

        saved_any = False
        dataframe_list: OrderedDict = OrderedDict(
            [(key, []) for key in data[0]]
        )
        for data_dict in data:
            for key, data_values in data_dict.items():
                df = construct_dataframe(data_values)

                if self.any_pulsemap_is_non_empty(data_dict) and len(df) > 0:
                    # only include data_dict in temp. databases if at least one pulsemap is non-empty,
                    # and the current extractor (df) is also non-empty (also since truth is always non-empty)
                    dataframe_list[key].append(df)

        dataframe = OrderedDict(
            [
                (
                    key,
                    pd.concat(dfs, ignore_index=True, sort=True)
                    if dfs
                    else pd.DataFrame(),
                )
                for key, dfs in dataframe_list.items()
            ]
        )
        # Can delete dataframe_list here to free up memory.

        # Save each dataframe to SQLite database
        self.debug(f"Saving to {output_file}")
        for table, df in dataframe.items():
            if len(df) > 0:
                create_table_and_save_to_sql(
                    df,
                    table,
                    output_file,
                    default_type="FLOAT",
                    integer_primary_key=not (
                        is_pulse_map(table) or is_mc_tree(table)
                    ),
                )
                saved_any = True

        if saved_any:
            self.debug("- Done saving")
        else:
            self.warning(f"No data saved to {output_file}")

    def merge_files(
        self,
        output_file: str,
        input_files: Optional[List[str]] = None,
        max_table_size: Optional[int] = None,
    ) -> None:
        """SQLite-specific method for merging output files/databases.

        Args:
            output_file: Name of the output file containing the merged results.
            input_files: Intermediate files/databases to be merged, according
                to the specific implementation. Default to None, meaning that
                all files/databases output by the current instance are merged.
            max_table_size: The maximum number of rows in any given table.
                If any one table exceed this limit, a new database will be
                created.
        """
        if max_table_size:
            self.warning(
                f"Merging got max_table_size of {max_table_size}. Will attempt to create databases with a maximum row count of this size."
            )
        self.max_table_size = max_table_size
        self._partition_count = 1

        if input_files is None:
            self.info("Merging files output by current instance.")
            self._input_files = self._output_files
        else:
            self._input_files = input_files

        if not output_file.endswith("." + self.file_suffix):
            output_file = ".".join([output_file, self.file_suffix])

        if os.path.exists(output_file):
            self.warning(
                f"Target path for merged database, {output_file}, already exists."
            )

        if len(self._input_files) > 0:
            self.info(f"Merging {len(self._input_files)} database files")
            # Create one empty database table for each extraction
            self._merged_table_names = self._extract_table_names(
                self._input_files
            )
            if self.max_table_size:
                output_file = self._adjust_output_file_name(output_file)
            self._create_empty_tables(output_file)
            self._row_counts = self._initialize_row_counts()
            # Merge temporary databases into newly created one
            self._merge_temporary_databases(output_file, self._input_files)
        else:
            self.warning("No temporary database files found!")

    # Internal methods
    def _adjust_output_file_name(self, output_file: str) -> str:
        if "_part_" in output_file:
            root = (
                output_file.split("_part_")[0]
                + output_file.split("_part_")[1][1:]
            )
        else:
            root = output_file
        str_list = root.split(".db")
        return str_list[0] + f"_part_{self._partition_count}" + ".db"

    def _update_row_counts(
        self, results: "OrderedDict[str, pd.DataFrame]"
    ) -> None:
        for table_name, data in results.items():
            self._row_counts[table_name] += len(data)
        return

    def _initialize_row_counts(self) -> Dict[str, int]:
        """Build dictionary with row counts. Initialized with 0.

        Returns:
            Dictionary where every field is a table name that contains
            corresponding row counts.
        """
        row_counts = {}
        for table_name in self._merged_table_names:
            row_counts[table_name] = 0
        return row_counts

    def _create_empty_tables(self, output_file: str) -> None:
        """Create tables for output database.

        Args:
            output_file: Path to database.
        """
        for table_name in self._merged_table_names:
            column_names = self._extract_column_names(
                self._input_files, table_name
            )
            if len(column_names) > 1:
                create_table(
                    column_names,
                    table_name,
                    output_file,
                    default_type="FLOAT",
                    integer_primary_key=not (
                        is_pulse_map(table_name) or is_mc_tree(table_name)
                    ),
                )

    def _get_tables_in_database(self, db: str) -> Tuple[str, ...]:
        with sqlite3.connect(db) as conn:
            table_names = tuple(
                [
                    p[0]
                    for p in (
                        conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        ).fetchall()
                    )
                ]
            )
        return table_names

    def _extract_table_names(
        self, db: Union[str, List[str]]
    ) -> Tuple[str, ...]:
        """Get the names of all tables in database `db`."""
        if isinstance(db, str):
            db = [db]
        results = [self._get_tables_in_database(path) for path in db]
        # @TODO: Check...
        if all([results[0] == r for r in results]):
            return results[0]
        else:
            unique_tables = []
            for tables in results:
                for table in tables:
                    if table not in unique_tables:
                        unique_tables.append(table)
            return tuple(unique_tables)

    def _extract_column_names(
        self, db_paths: List[str], table_name: str
    ) -> List[str]:
        for db_path in db_paths:
            tables_in_database = self._get_tables_in_database(db_path)
            if table_name in tables_in_database:
                with sqlite3.connect(db_path) as con:
                    query = f"select * from {table_name} limit 1"
                    columns = pd.read_sql(query, con).columns
                if len(columns):
                    return columns
        return []

    def any_pulsemap_is_non_empty(self, data_dict: Dict[str, Dict]) -> bool:
        """Check whether there are non-empty pulsemaps extracted from P frame.

        Takes in the data extracted from the P frame, then retrieves the
        values, if there are any, from the pulsemap key(s) (e.g
        SplitInIcePulses). If at least one of the pulsemaps is non-empty then
        return true. If no pulsemaps exist, i.e., if no `I3FeatureExtractor` is
        called e.g. because `I3GenericExtractor` is used instead, always return
        True.
        """
        if len(self._pulsemaps) == 0:
            return True

        pulsemap_dicts = [data_dict[pulsemap] for pulsemap in self._pulsemaps]
        return any(d["dom_x"] for d in pulsemap_dicts)

    def _submit_to_database(
        self, database: str, key: str, data: pd.DataFrame
    ) -> None:
        """Submit data to the database with specified key."""
        if len(data) == 0:
            self.info(f"No data provided for {key}.")
            return
        engine = sqlalchemy.create_engine("sqlite:///" + database)
        data.to_sql(key, engine, index=False, if_exists="append")
        engine.dispose()

    def _extract_everything(self, db: str) -> "OrderedDict[str, pd.DataFrame]":
        """Extract everything from the temporary database `db`.

        Args:
            db: Path to temporary database.

        Returns:
            Dictionary containing the data for each extracted table.
        """
        results = OrderedDict()
        table_names = self._extract_table_names(db)
        with sqlite3.connect(db) as conn:
            for table_name in table_names:
                query = f"select * from {table_name}"
                try:
                    data = pd.read_sql(query, conn)
                except:  # noqa: E722
                    data = []
                results[table_name] = data
        return results

    def _merge_temporary_databases(
        self,
        output_file: str,
        input_files: List[str],
    ) -> None:
        """Merge the temporary databases.

        Args:
            output_file: path to the final database
            input_files: list of names of temporary databases
        """
        file_count = 0
        for input_file in tqdm(input_files, colour="green"):
            results = self._extract_everything(input_file)
            for table_name, data in results.items():
                self._submit_to_database(output_file, table_name, data)
            file_count += 1
            if (self.max_table_size is not None) & (
                file_count < len(input_files)
            ):
                self._update_row_counts(results)
                maximum_row_count_reached = False
                for table in self._row_counts.keys():
                    assert self.max_table_size is not None
                    if self._row_counts[table] >= self.max_table_size:
                        maximum_row_count_reached = True
                if maximum_row_count_reached:
                    self._partition_count += 1
                    output_file = self._adjust_output_file_name(output_file)
                    self.info(
                        f"Maximum row count reached. Creating new partition at {output_file}"
                    )
                    self._create_empty_tables(output_file)
                    self._row_counts = self._initialize_row_counts()


# Implementation-specific utility function(s)
def construct_dataframe(extraction: Dict[str, Any]) -> pd.DataFrame:
    """Convert extraction to pandas.DataFrame.

    Args:
        extraction: Dictionary with the extracted data.

    Returns:
        Extraction as pandas.DataFrame.
    """
    all_scalars = True
    for value in extraction.values():
        if isinstance(value, (list, tuple, dict)):
            all_scalars = False
            break

    out = pd.DataFrame(extraction, index=[0] if all_scalars else None)
    return out


def is_pulse_map(table_name: str) -> bool:
    """Check whether `table_name` corresponds to a pulse map."""
    return "pulse" in table_name.lower() or "series" in table_name.lower()


def is_mc_tree(table_name: str) -> bool:
    """Check whether `table_name` corresponds to an MC tree."""
    return "I3MCTree" in table_name
