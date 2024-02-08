"""DataConverter for the Parquet backend."""

import os
from typing import List, Optional, Dict

import awkward
import pandas as pd

from .graphnet_writer import GraphNeTWriter


class ParquetWriter(GraphNeTWriter):
    """Class for writing interim data format to Parquet."""

    # Class variables
    file_suffix: str = ".parquet"

    # Abstract method implementation(s)
    def _save_file(
        self,
        data: Dict[str, pd.DataFrame],
        output_file_path: str,
        n_events: int,
    ) -> None:
        """Save data to parquet file."""
        # Check(s)
        if os.path.exists(output_file_path):
            self.warning(
                f"Output file {output_file_path} already exists. Overwriting."
            )

        self.debug(f"Saving to {output_file_path}")
        awkward.to_parquet(awkward.from_iter(data), output_file_path)
        self.debug("- Done saving")
