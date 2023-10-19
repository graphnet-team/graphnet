"""Prometheus-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import PROMETHEUS_GEOMETRY_TABLE_DIR


class ORCA150(Detector):
    """`Detector` class for Prometheus prototype."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca150.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350) / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class Prometheus(ORCA150):
    """Reference to ORCA150."""
