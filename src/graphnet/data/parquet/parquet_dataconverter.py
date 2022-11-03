"""DataConverter for the Parquet backend."""

from collections import OrderedDict
import os
from typing import List, Optional

import awkward

from graphnet.data.dataconverter import DataConverter  # type: ignore[attr-defined]


class ParquetDataConverter(DataConverter):
    """Class for converting I3-files to Parquet format."""

    # Class variables
    file_suffix: str = "parquet"

    # Abstract method implementation(s)
    def save_data(self, data: List[OrderedDict], output_file: str) -> None:
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
    ) -> None:
        """Parquet-specific method for merging output files.

        Args:
            output_file: Name of the output file containing the merged results.
            input_files: Intermediate files to be merged, according to the
                specific implementation. Default to None, meaning that all
                files output by the current instance are merged.

        Raises:
            NotImplementedError: If the method has not been implemented for the
                Parquet backend.
        """
        raise NotImplementedError()
