from collections import OrderedDict
import os
from typing import List, Optional

import awkward

from graphnet.data.dataconverter import DataConverter


class ParquetDataConverter(DataConverter):

    # Class variables
    file_suffix = "parquet"

    # Abstract method implementation(s)
    def save_data(self, data: List[OrderedDict], output_file: str):
        """Save data to parquet file."""
        # Check(s)
        if os.path.exists(output_file):
            self.warning(
                f"Output file {output_file} already exists. Overwriting."
            )

        self.debug(f"Saving to {output_file}")
        self.debug(
            f"- Data has {len(data)} events and {len(data[0])} tables for each"
        )

        awkward.to_parquet(awkward.from_iter(data), output_file)

        self.debug("- Done saving")
        self._output_files.append(output_file)

    def merge_files(
        self, output_file: str, input_files: Optional[List[str]] = None
    ):
        raise NotImplementedError()
