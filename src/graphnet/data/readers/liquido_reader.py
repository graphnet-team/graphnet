"""Modules for reading data files from LiquidO."""

from typing import List, Union, Dict
from glob import glob
import os
import pandas as pd

from graphnet.data.extractors.liquido import H5Extractor
from .graphnet_file_reader import GraphNeTFileReader


class LiquidOReader(GraphNeTFileReader):
    """A class for reading h5 files from LiquidO."""

    _accepted_file_extensions = [".h5"]
    _accepted_extractors = [H5Extractor]

    def __call__(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Extract data from single parquet file.

        Args:
            file_path: Path to h5 file.

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
        """Search folder(s) for h5 files.

        Args:
            path: directory to search for h5 files.

        Returns:
            List of h5 files in the folders.
        """
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.h5")))
        return files
