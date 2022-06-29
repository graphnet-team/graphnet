from collections import OrderedDict
import os
from typing import List

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
            self.logger.warning(
                f"Output file {output_file} already exists. Overwriting."
            )

        self.logger.debug(f"Saving to {output_file}")
        self.logger.debug(
            f"- Data has {len(data)} events and {len(data[0])} tables for each"
        )

        awkward.to_parquet(awkward.from_iter(data), output_file)

        self.logger.debug("- Done saving")

    def merge_files(self, input_files: List[str], output_file: str):
        raise NotImplementedError()
