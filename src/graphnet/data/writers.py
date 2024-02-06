"""Module containing `GraphNeTFileSaveMethod`(s).

These modules are used to save the interim data format from `DataConverter` to
a deep-learning friendly file format.
"""

import os
from typing import List, Union, Dict, Any, OrderedDict
from abc import abstractmethod, ABC

from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger
from graphnet.data.sqlite.sqlite_utilities import (
    create_table,
    create_table_and_save_to_sql,
)

import pandas as pd


class GraphNeTFileSaveMethod(Logger, ABC):
    """Generic base class for saving interim data format in `DataConverter`.

    Classes inheriting from `GraphNeTFileSaveMethod` must implement the
    `save_file` method, which recieves the interim data format from
    from a single file.

    In addition, classes inheriting from `GraphNeTFileSaveMethod` must
    set the `file_extension` property.
    """

    @abstractmethod
    def _save_file(
        self,
        data: Dict[str, pd.DataFrame],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save the interim data format from a single input file.

        Args:
            data: the interim data from a single input file.
            output_file_path: output file path.
            n_events: Number of events container in `data`.
        """

    @final
    def __call__(
        self,
        data: Dict[str, pd.DataFrame],
        file_name: str,
        output_dir: str,
        n_events: int,
    ) -> None:
        """Save data.

        Args:
            data: data to be saved.
            file_name: name of input file. Will be used to generate output
                        file name.
            output_dir: directory to save data to.
            n_events: Number of events in `data`.
        """
        # make dir
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = (
            os.path.join(output_dir, file_name) + self.file_extension
        )

        self._save_file(
            data=data, output_file_path=output_file_path, n_events=n_events
        )
        return

    @property
    def file_extension(self) -> str:
        """Return file extension used to store the data."""
        return self._file_extension  # type: ignore


class SQLiteSaveMethod(GraphNeTFileSaveMethod):
    """A method for saving GraphNeT's interim dataformat to SQLite."""

    _file_extension = ".db"

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
