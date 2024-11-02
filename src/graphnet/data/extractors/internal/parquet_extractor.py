"""Parquet Extractor for conversion from internal parquet format."""

import polars as pol
import pandas as pd

from graphnet.data.extractors import Extractor


class ParquetExtractor(Extractor):
    """Class for extracting information from internal GraphNeT parquet files.

    Contains functionality required to extract data from internal parquet
    files, i.e files saved using the ParquetWriter. This allows for conversion
    between internal data formats.
    """

    def __init__(self, extractor_name: str):
        """Construct ParquetExtractor.

        Args:
            extractor_name: Name of the `ParquetExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
        """
        # Member variable(s)
        self._table = extractor_name
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, file_path: str) -> pd.DataFrame:
        """Extract information from parquet file."""
        if self._table in file_path:
            return pol.read_parquet(file_path).to_pandas()
        else:
            return None
