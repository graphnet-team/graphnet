"""Modules for reading data files from the Prometheus project."""

from typing import List, Union, OrderedDict
import pandas as pd
from pathlib import Path

from graphnet.data.extractors.prometheus import PrometheusExtractor
from .graphnet_file_reader import GraphNeTFileReader


class PrometheusReader(GraphNeTFileReader):
    """A class for reading parquet files from Prometheus simulation."""

    _accepted_file_extensions = [".parquet"]
    _accepted_extractors = [PrometheusExtractor]

    def __call__(self, file_path: str) -> List[OrderedDict]:
        """Extract data from single parquet file.

        Args:
            file_path: Path to parquet file.

        Returns:
            Extracted data.
        """
        # Open file
        outputs = []
        file = pd.read_parquet(file_path)
        for k in range(len(file)):  # Loop over events in file
            extracted_event = OrderedDict()
            for extractor in self._extractors:
                assert isinstance(extractor, PrometheusExtractor)
                if extractor._table in file.columns:
                    output = extractor(file[extractor._table][k])
                    extracted_event[extractor._extractor_name] = output
            outputs.append(extracted_event)
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for parquet files.

        Args:
            path: directory to search for parquet files.

        Returns:
            List of parquet files in the folders.
        """
        files = []
        if isinstance(path, str):
            path = [path]

        # List of files as Path objects
        for p in path:
            files.extend(
                list(Path(p).rglob(f"*{self.accepted_file_extensions}"))
            )

        # List of files as str's
        paths_as_str: List[str] = []
        for f in files:
            paths_as_str.append(f.absolute().as_posix())

        return paths_as_str
