"""Module containing different reader for GraphNeT internal parquet format."""

from typing import List, Union, Dict
from glob import glob
import os
import pandas as pd

from graphnet.data.extractors.internal import ParquetExtractor
from .graphnet_file_reader import GraphNeTFileReader


class ParquetReader(GraphNeTFileReader):
    """A class for reading the internal GraphNeT parquet format."""

    _accepted_file_extensions = [".parquet"]
    _accepted_extractors = [ParquetExtractor]

    def __call__(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Extract data from single parquet file.

        Args:
            file_path: Path to parquet file.

        Returns:
            Extracted data.
        """
        # Open file
        outputs = {}
        for extractor in self._extractors:
            output = extractor(file_path)
            if output is not None:
                outputs[extractor._extractor_name] = output
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search parquet folders for files.

        Args:
            path: directory containing the parquet folders.

        Returns:
            List of parquet files in the folders.
        """
        # Find all I3 and GCD files in the specified directories.
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            for extractor in self._extractors:
                files.extend(
                    glob(
                        os.path.join(p, extractor._extractor_name, "*.parquet")
                    )
                )
        return files
