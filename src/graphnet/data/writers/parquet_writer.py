"""DataConverter for the Parquet backend."""

import os
from typing import List, Optional, Dict

import awkward
import pandas as pd

from .graphnet_writer import GraphNeTWriter


class ParquetWriter(GraphNeTWriter):
    """Class for writing interim data format to Parquet."""

    # Class variables
    _file_extension = ".parquet"
    _merge_dataframes = False

    # Abstract method implementation(s)
    def _save_file(
        self,
        data: Dict[str, List[pd.DataFrame]],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to parquet."""
        # Check(s)

        if n_events > 0:
            events = []
            for k in range(n_events):
                event = {}
                for table in data.keys():
                    event[table] = data[table][k].to_dict(orient="list")

                events.append(event)

            awkward.to_parquet(awkward.from_iter(events), output_file_path)

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
