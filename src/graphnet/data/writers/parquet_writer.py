"""DataConverter for the Parquet backend."""

import os
from typing import List, Optional, Dict

import awkward
import pandas as pd

from .graphnet_writer import GraphNeTWriter


class ParquetWriter(GraphNeTWriter):
    """Class for writing interim data format to Parquet."""

    def __init__(self, index_column: str = "event_no") -> None:
        """Construct `ParquetWriter`.

        Args:
            index_column: The column used for indexation.
                             Defaults to "event_no".
        """
        # Class variables
        self._file_extension = ".parquet"
        self._merge_dataframes = True
        self._index_column = index_column

    # Abstract method implementation(s)
    def _save_file(
        self,
        data: Dict[str, pd.DataFrame],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to parquet."""
        # Check(s)

        if n_events > 0:
            for table in data.keys():
                save_path = os.path.dirname(output_file_path)
                file_name = os.path.splitext(
                    os.path.basename(output_file_path)
                )[0]

                table_dir = os.path.join(save_path, f"{table}")
                os.makedirs(table_dir, exist_ok=True)
                df = data[table].set_index(self._index_column)
                df.to_parquet(
                    os.path.join(table_dir, file_name + f"_{table}.parquet")
                )

    def merge_files(self, files: List[str], output_dir: str) -> None:
        """Merge parquet files.

        Args:
            files: input files for merging.
            output_dir: directory to store merged file(s) in.

        Raises:
            NotImplementedError
        """
        self.error(f"{self.__class__.__name__} does not have a merge method.")
        raise NotImplementedError
