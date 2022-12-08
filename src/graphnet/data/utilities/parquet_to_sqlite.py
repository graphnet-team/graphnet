"""Utilities for converting files from Parquet to SQLite."""

import glob
import os
from typing import List, Optional, Union

import awkward as ak
import numpy as np
import pandas as pd
from tqdm.auto import trange

from graphnet.data.sqlite.sqlite_utilities import create_table_and_save_to_sql
from graphnet.utilities.logging import LoggerMixin


class ParquetToSQLiteConverter(LoggerMixin):
    """Convert Parquet files to a SQLite database.

    Each event in the parquet file(s) are assigned a unique event id. By
    default, every field in the parquet file(s) are extracted. One can choose
    to exclude certain fields by using the argument exclude_fields.
    """

    def __init__(
        self,
        parquet_path: Union[str, List[str]],
        mc_truth_table: str = "mc_truth",
        excluded_fields: Optional[Union[str, List[str]]] = None,
    ):
        """Construct `ParquetToSQLiteConverter`."""
        # checks
        if isinstance(parquet_path, str):
            pass
        elif isinstance(parquet_path, list):
            assert isinstance(
                parquet_path[0], str
            ), "Argument `parquet_path` must be a string or list of strings"
        else:
            assert isinstance(
                parquet_path, str
            ), "Argument `parquet_path` must be a string or list of strings"

        assert isinstance(
            mc_truth_table, str
        ), "Argument `mc_truth_table` must be a string"
        self._parquet_files = self._find_parquet_files(parquet_path)
        if excluded_fields is not None:
            self._excluded_fields = excluded_fields
        else:
            self._excluded_fields = []
        self._mc_truth_table = mc_truth_table
        self._event_counter = 0

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

    def run(self, outdir: str, database_name: str) -> None:
        """Run Parquet to SQLite conversion.

        Args:
            outdir: Output directory for SQLite database.
            database_name: Name of output SQLite database.
        """
        self._create_output_directories(outdir, database_name)
        database_path = os.path.join(
            outdir, database_name, "data", database_name + ".db"
        )
        self.info(f"Processing {len(self._parquet_files)} Parquet file(s)")
        for i in trange(
            len(self._parquet_files),
            unit="file(s)",
            colour="green",
            position=0,
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
                        database_path,
                        parquet_file,
                        parquet_file.fields[j],
                        n_events_in_file,
                    )
            self._event_counter += n_events_in_file
        self._save_config(outdir, database_name)
        self.info(
            "Database saved at: \n"
            f"{outdir}/{database_name}/data/{database_name}.db"
        )

    def _count_events(self, open_parquet_file: ak.Array) -> int:
        return len(open_parquet_file[self._mc_truth_table])

    def _save_to_sql(
        self,
        database_path: str,
        ak_array: ak.Array,
        field_name: str,
        n_events_in_file: int,
    ) -> None:
        df = self._convert_to_dataframe(ak_array, field_name, n_events_in_file)

        if len(df) > n_events_in_file:
            is_pulse_map = True
        else:
            is_pulse_map = False

        create_table_and_save_to_sql(
            df,
            field_name,
            database_path,
            integer_primary_key=not is_pulse_map,
        )

    def _convert_to_dataframe(
        self,
        ak_array: ak.Array,
        field_name: str,
        n_events_in_file: int,
    ) -> pd.DataFrame:
        df = pd.DataFrame(ak.to_pandas(ak_array[field_name]))
        if len(df.columns) == 1:
            if df.columns == ["values"]:
                df.columns = [field_name]

        # If true, the dataframe contains more than 1 row pr. event (i.e.,
        # pulsemap).
        if len(df) != n_events_in_file:
            event_nos = []
            c = 0
            for event_no in range(
                self._event_counter, self._event_counter + n_events_in_file, 1
            ):
                try:
                    event_nos.extend(
                        np.repeat(event_no, len(df[df.columns[0]][c])).tolist()
                    )

                # KeyError indicates that this df has no entry for event_no
                # (e.g., an event with no detector response).
                except KeyError:
                    pass
                c += 1
        else:
            event_nos = np.arange(0, n_events_in_file, 1) + self._event_counter
        df["event_no"] = event_nos
        return df

    def _create_output_directories(
        self, outdir: str, database_name: str
    ) -> None:
        os.makedirs(outdir + "/" + database_name + "/data", exist_ok=True)
        os.makedirs(outdir + "/" + database_name + "/config", exist_ok=True)

    def _save_config(self, outdir: str, database_name: str) -> None:
        """Save the list of converted Parquet files to a CSV file."""
        df = pd.DataFrame(data=self._parquet_files, columns=["files"])
        df.to_csv(outdir + "/" + database_name + "/config/files.csv")
