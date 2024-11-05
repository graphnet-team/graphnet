"""H5 Extractor for LiquidO data files."""

from typing import List
import numpy as np
import pandas as pd
import h5py

from graphnet.data.extractors import Extractor


class H5Extractor(Extractor):
    """Class for extracting information from LiquidO h5 files."""

    def __init__(self, extractor_name: str, column_names: List[str]):
        """Construct H5Extractor.

        Args:
            extractor_name: Name of the `ParquetExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
            column_names: Name of the columns in `extractor_name`.
        """
        # Member variable(s)
        self._table = extractor_name
        self._column_names = column_names
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, file_path: str) -> pd.DataFrame:
        """Extract information from h5 file."""
        with h5py.File(file_path, "r") as f:
            available_tables = [f for f in f.keys()]
            if self._table in available_tables:
                array = f[self._table][:]
                # Will throw error if the number of columns don't match
                self._verify_columns(array)
                df = pd.DataFrame(array, columns=self._column_names)
                return df
            else:
                return None

    def _verify_columns(self, array: np.ndarray) -> None:
        try:
            assert array.shape[1] == len(self._column_names)
        except AssertionError as e:
            self.error(
                f"Got {len(self._column_names)} column names but "
                f"{self._table} has {array.shape[1]}. Please make sure "
                f"that the column names match. ({self._column_names})"
            )
            raise e


class H5HitExtractor(H5Extractor):
    """Extractor for `HitData` in LiquidO H5 files."""

    def __init__(self) -> None:
        """Extractor for `HitData` in LiquidO H5 files."""
        # Base class constructor
        super().__init__(
            extractor_name="HitData",
            column_names=[
                "event_no",
                "sipmID",
                "sipm_x",
                "sipm_y",
                "sipm_z",
                "t",
                "var",
            ],
        )


class H5TruthExtractor(H5Extractor):
    """Extractor for `TruthData` in LiquidO H5 files."""

    def __init__(self) -> None:
        """Extractor for `TruthData` in LiquidO H5 files."""
        # Base class constructor
        super().__init__(
            extractor_name="TruthData",
            column_names=[
                "event_no",
                "vertex_x",
                "vertex_y",
                "vertex_z",
                "zenith",
                "azimuth",
                "interaction_time",
                "energy",
                "pid",
            ],
        )
