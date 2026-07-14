"""Parquet Extractor for conversion of simulation files from PROMETHEUS."""

from typing import List, Optional
import pandas as pd
import numpy as np

from graphnet.data.constants import TRUTH
from graphnet.data.extractors import Extractor


class PrometheusExtractor(Extractor):
    """Class for extracting information from PROMETHEUS parquet files.

    Contains functionality required to extract data from PROMETHEUS
    parquet files.
    """

    def __init__(self, extractor_name: str, columns: List[str]):
        """Construct PrometheusExtractor.

        Args:
            extractor_name: Name of the `PrometheusExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
            columns: List of column names to extract from the table.
        """
        # Member variable(s)
        self._table = extractor_name
        self._columns = columns
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, event: pd.DataFrame) -> pd.DataFrame:
        """Extract information from parquet file."""
        output = {key: [] for key in self._columns}  # type: ignore
        for key in self._columns:
            if key in event.keys():
                data = event[key]
                if isinstance(data, np.ndarray):
                    data = data.tolist()
                if isinstance(data, list):
                    output[key].extend(data)
                else:
                    output[key].append(data)
            else:
                self.warning_once(f"{key} not found in {self._table}!")
        return output


class PrometheusTruthExtractor(PrometheusExtractor):
    """Class for extracting event level truth from Prometheus parquet files.

    This Extractor will "initial_state" i.e. neutrino truth.
    """

    def __init__(
        self,
        table_name: str = "mc_truth",
        columns: Optional[List[str]] = None,
    ) -> None:
        """Construct PrometheusTruthExtractor.

        Args:
            table_name: Name of the table in the parquet files that contain
                event-level truth. Defaults to "mc_truth".
            columns: Columns to extract from the table. Defaults to
                `TRUTH.PROMETHEUS`, the truth fields written by every
                Prometheus injection type. Override to include truth-level
                data not present per default, e.g. LeptonInjector's
                "bjorken_x", "bjorken_y" and "column_depth".
        """
        if columns is None:
            columns = list(TRUTH.PROMETHEUS)
        super().__init__(extractor_name=table_name, columns=columns)


class PrometheusFeatureExtractor(PrometheusExtractor):
    """Class for extracting pulses/photons from Prometheus parquet files."""

    def __init__(self, table_name: str = "photons"):
        """Construct PrometheusFeatureExtractor.

        Args:
            table_name: Name of table in parquet files that contain the
                photons/pulses. Defaults to "photons".
        """
        columns = [
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "string_id",
            "sensor_id",
            "t",
        ]
        super().__init__(extractor_name=table_name, columns=columns)
