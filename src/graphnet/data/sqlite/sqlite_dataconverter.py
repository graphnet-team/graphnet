from collections import OrderedDict, Iterable
import json
import numpy as np
import os
import pandas as pd
import sqlalchemy
import sqlite3
from tqdm import tqdm
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

from graphnet.data.dataconverter import DataConverter
from graphnet.data.sqlite.sqlite_utilities import run_sql_code, save_to_sql


class SQLiteDataConverter(DataConverter):

    # Class variables
    file_suffix = "db"

    # Abstract method implementation(s)
    def save_data(self, data: List[OrderedDict], output_file: str):
        """Save data to SQLite database."""
        # Check(s)
        if os.path.exists(output_file):
            self.logger.warning(
                f"Output file {output_file} already exists. Appending."
            )

        # Concatenate data
        dataframe = OrderedDict()
        for data_dict in data:
            for key, data in data_dict.items():
                df = construct_dataframe(data)

                if self.any_pulsemap_is_non_empty(data_dict) and len(df) > 0:
                    # only include data_dict in temp. databases if at least one pulsemap is non-empty,
                    # and the current extractor (df) is also non-empty (also since truth is always non-empty)
                    if key in dataframe:
                        dataframe[key] = dataframe[key].append(
                            df, ignore_index=True, sort=True
                        )
                    else:
                        dataframe[key] = df

        # Save each dataframe to SQLite database
        self.logger.debug(f"Saving to {output_file}")
        saved_any = False
        for table, df in dataframe.items():
            if len(df) > 0:
                save_to_sql(df, table, output_file)
                saved_any = True

        if saved_any:
            self.logger.debug("- Done saving")
        else:
            self.logger.warning(f"No data saved to {output_file}")

    def merge_files(
        self, output_file: str, input_files: Optional[List[str]] = None
    ):
        if input_files is None:
            self.logger.info("Merging files output by current instance.")
            input_files = self._output_files

        if not output_file.endswith("." + self.file_suffix):
            output_file = ".".join([output_file, self.file_suffix])

        if os.path.exists(output_file):
            self.logger.warning(
                f"Target path for merged database, {output_file}, already exists."
            )

        if len(input_files) > 0:
            self.logger.info(f"Merging {len(input_files)} database files")

            # Create one empty database table for each extraction
            table_names = self._extract_table_names(input_files)
            for table_name in table_names:
                column_names = self._extract_column_names(
                    input_files, table_name
                )
                if len(column_names) > 1:
                    is_pulse_map = is_pulsemap_check(table_name)
                    self._create_table(
                        output_file,
                        table_name,
                        column_names,
                        is_pulse_map=is_pulse_map,
                    )

            # Merge temporary databases into newly created one
            self._merge_temporary_databases(output_file, input_files)
        else:
            self.logger.warning("No temporary database files found!")

    # Internal methods
    def _extract_table_names(self, db: Union[str, List[str]]) -> Tuple[str]:
        """Get the names of all tables in database `db`."""
        if isinstance(db, list):
            results = [self._extract_table_names(path) for path in db]
            # @TODO: Check...
            assert all([results[0] == r for r in results])
            return results[0]

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

    def _extract_column_names(self, db_paths, table_name):
        for db_path in db_paths:
            with sqlite3.connect(db_path) as con:
                query = f"select * from {table_name} limit 1"
                columns = pd.read_sql(query, con).columns
            if len(columns):
                return columns
        return []

    def any_pulsemap_is_non_empty(self, data_dict: OrderedDict) -> bool:
        """Check whether there are non-empty pulsemaps extracted from P frame.

        Takes in the data extracted from the P frame, then retrieves the
        values, if there are any, from the pulsemap key(s)
        (e.g SplitInIcePulses). If at least one of the pulsemaps is non-empty
        then return true. If no pulsemaps exist, i.e., if no
        `I3FeatureExtractor` is called e.g. because `I3GenericExtractor` is
        used instead, always return True.
        """
        if len(self._pulsemaps) == 0:
            return True

        pulsemap_dicts = map(data_dict.get, self._pulsemaps)
        return any(d["dom_x"] for d in pulsemap_dicts)

    def _attach_index(self, database: str, table_name: str):
        """Attaches the table index. Important for query times!"""
        code = (
            "PRAGMA foreign_keys=off;\n"
            "BEGIN TRANSACTION;\n"
            f"CREATE INDEX event_no_{table_name} ON {table_name} (event_no);\n"
            "COMMIT TRANSACTION;\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(database, code)

    def _create_table(self, database, table_name, columns, is_pulse_map=False):
        """Creates a table.

        Args:
            database (str): path to the database
            table_name (str): name of the table
            columns (str): the names of the columns of the table
            is_pulse_map (bool, optional): whether or not this is a pulse map table. Defaults to False.
        """
        query_columns = list()
        for column in columns:
            if column == "event_no":
                if not is_pulse_map:
                    type_ = "INTEGER PRIMARY KEY NOT NULL"
                else:
                    type_ = "NOT NULL"
            else:
                type_ = "FLOAT"
            query_columns.append(f"{column} {type_}")
        query_columns = ", ".join(query_columns)

        code = (
            "PRAGMA foreign_keys=off;\n"
            f"CREATE TABLE {table_name} ({query_columns});\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(database, code)

        if is_pulse_map:
            self.logger.debug(table_name)
            self.logger.debug("Attaching indices")
            self._attach_index(database, table_name)
        return

    def _submit_to_database(self, database: str, key: str, data: pd.DataFrame):
        """Submits data to the database with specified key."""
        if len(data) == 0:
            if self._verbose:
                self.logger.info(f"No data provided for {key}.")
            return
        engine = sqlalchemy.create_engine("sqlite:///" + database)
        data.to_sql(key, engine, index=False, if_exists="append")
        engine.dispose()

    def _extract_everything(self, db: str) -> "OrderedDict[str, pd.DataFrame]":
        """Extracts everything from the temporary database `db`.

        Args:
            db (str): Path to temporary database

        Returns:
            results (dict): Contains the data for each extracted table
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
    ):
        """Merges the temporary databases.

        Args:
            output_file (str): path to the final database
            input_files (list): list of names of temporary databases
        """
        for input_file in tqdm(input_files, colour="green"):
            results = self._extract_everything(input_file)
            for table_name, data in results.items():
                self._submit_to_database(output_file, table_name, data)


# Implementation-specific utility function(s)
def construct_dataframe(extraction: Dict[str, Any]) -> pd.DataFrame:
    """Converts extraction to pandas.DataFrame.

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


def is_pulsemap_check(table_name: str) -> bool:
    """Check whether `table_name` corresponds to a pulsemap, and not a truth or RETRO table."""
    if "retro" in table_name.lower() or "truth" in table_name.lower():
        return False
    else:  # Could have to include the lower case word 'pulse'?
        return True
