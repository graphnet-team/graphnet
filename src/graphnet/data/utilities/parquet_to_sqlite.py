from dataclasses import field
import pandas as pd
import os
import sqlite3
import awkward as ak
import awkward
import glob
from typing import List, Optional, Union
from tqdm.auto import trange
import numpy as np
import sqlalchemy
from graphnet.data.sqlite.sqlite_utilities import run_sql_code


class ParquetToSQLiteConverter:
    """Converts Parquet files to a SQLite database. Each event in the parquet file(s) are assigned a unique event id.
    By default, every field in the parquet file(s) are extracted. One can choose to exclude certain fields by using the argument exclude_fields.
    """

    def __init__(
        self,
        mc_truth_table: str = "mc_truth",
        parquet_path: Union[str, List[str]] = None,
        excluded_fields: Optional[Union[str, List[str]]] = None,
    ):
        # checks
        assert isinstance(
            parquet_path, str
        ), "Argument `parquet_path` must be a string or list of strings"
        assert isinstance(
            mc_truth_table, str
        ), "Argument `mc_truth_table` must be a string"
        self._parquet_files = self._find_files(parquet_path)
        if excluded_fields is not None:
            self._excluded_fields = excluded_fields
        else:
            self._excluded_fields = []
        self._mc_truth_table = mc_truth_table
        self._event_counter = 0
        self._created_tables = []

    def _find_parquet_files(self, paths: Union[str, List[str]]) -> List[str]:
        if isinstance(paths, str):
            if paths.endswith(".parquet"):
                files = [paths]
            else:
                files = glob.glob(f"{paths}/*.parquet")
        elif isinstance(paths, list):
            files = []
            for path in paths:
                files.extend(self._find_parquet_files(path))
        assert len(files) > 0, f"No files found in {paths}"
        return files

    def run(self, outdir: str = None, database_name: str = None):
        self._setup_directory(outdir, database_name)
        for i in trange(
            len(self._parquet_files), desc="Main", colour="#0000ff", position=0
        ):
            parquet_file = ak.from_parquet(self._parquet_files[i])
            n_events_in_file = self._count_events(parquet_file)
            for j in trange(
                len(parquet_file.fields),
                desc="%s" % (self._parquet_files[i].split("/")[-1]),
                colour="#ffa500",
                position=1,
                leave=False,
            ):
                if parquet_file.fields[j] not in self._excluded_fields:
                    self._save_to_sql(
                        outdir,
                        database_name,
                        parquet_file,
                        parquet_file.fields[j],
                        n_events_in_file,
                    )
            self._event_counter += n_events_in_file
        self._save_config(outdir, database_name)
        print(
            f"Database saved at \n{outdir}/{database_name}/data/{database_name}.db"
        )

    def _count_events(self, open_parquet_file: str = None):
        return len(open_parquet_file[self._mc_truth_table])

    def _save_to_sql(
        self,
        outdir: str = None,
        database_name: str = None,
        ak_array: awkward.Array = None,
        field_name: str = None,
        n_events_in_file: int = None,
    ):
        df = self._make_df(ak_array, field_name, n_events_in_file)
        if field_name in self._created_tables:
            self._submit_to_database(outdir, database_name, field_name, df)
        else:
            self._create_table(
                df, field_name, outdir, database_name, n_events_in_file
            )
            self._created_tables.append(field_name)
            self._submit_to_database(outdir, database_name, field_name, df)

    def _make_df(
        self,
        ak_array: awkward.Array = None,
        field_name: str = None,
        n_events_in_file: int = None,
    ):
        df = pd.DataFrame(ak.to_pandas(ak_array[field_name]))
        if len(df.columns) == 1:
            if df.columns == ["values"]:
                df.columns = [field_name]
        if (
            len(df) != n_events_in_file
        ):  # if true, the dataframe contains more than 1 row pr. event (e.g. Pulsemap).
            event_nos = []
            for event_no in range(n_events_in_file + self._event_counter):
                try:
                    event_nos.extend(
                        np.repeat(
                            event_no, len(df[df.columns[0]][event_no])
                        ).tolist()
                    )
                except KeyError:  # KeyError indicates that this df has no entry for event_no (e.g. an event with no detector response)
                    pass
        else:
            event_nos = np.arange(0, n_events_in_file, 1) + self._event_counter
        df["event_no"] = event_nos
        return df

    def _submit_to_database(
        self, outdir: str, database_name: str, key: str, data: pd.DataFrame
    ):
        """Submits data to the database with specified key."""
        if len(data) == 0:
            if self._verbose:
                self.logger.info(f"No data provided for {key}.")
            return
        engine = sqlalchemy.create_engine(
            "sqlite:///"
            + outdir
            + "/"
            + database_name
            + "/data/"
            + database_name
            + ".db"
        )
        data.to_sql(key, engine, index=False, if_exists="append")
        engine.dispose()

    def _create_table(
        self,
        df: pd.DataFrame,
        field_name: str = None,
        outdir: str = None,
        database_name: str = None,
        n_events_in_file: int = None,
    ):
        """Creates a table.

        Args:
            database (str): path to the database
            table_name (str): name of the table
            columns (str): the names of the columns of the table
            is_pulse_map (bool, optional): whether or not this is a pulse map table. Defaults to False.
        """
        columns = df.columns
        if len(df) > n_events_in_file:
            is_pulse_map = True
        else:
            is_pulse_map = False
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
            f"CREATE TABLE {field_name} ({query_columns});\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(
            outdir + "/" + database_name + "/data/" + database_name + ".db",
            code,
        )
        print(is_pulse_map, field_name)
        if is_pulse_map:
            self._attach_index(
                outdir
                + "/"
                + database_name
                + "/data/"
                + database_name
                + ".db",
                table_name=field_name,
            )
        return

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
        return

    def _setup_directory(self, outdir: str = None, database_name: str = None):
        os.makedirs(outdir + "/" + database_name + "/data", exist_ok=True)
        os.makedirs(outdir + "/" + database_name + "/config", exist_ok=True)
        return

    def _save_config(self, outdir: str = None, database_name: str = None):
        df = pd.DataFrame(data=self._parquet_files, columns=["files"])
        df.to_csv(outdir + "/" + database_name + "/config/files.csv")
        return
