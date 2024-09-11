"""LiquidO-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import LIQUIDO_GEOMETRY_TABLE_DIR


class LiquidO_v1(Detector):
    """`Detector` class for LiquidO prototype."""

    geometry_table_path = os.path.join(
        LIQUIDO_GEOMETRY_TABLE_DIR, "liquido_v1.parquet"
    )
    xyz = ["sipm_x", "sipm_y", "sipm_z"]
    string_id_column = "fiber_id"
    sensor_id_column = "sipm_id"
    sensor_time_column = "t"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sipm_x": self._sipm_xyz,
            "sipm_y": self._sipm_xyz,
            "sipm_z": self._sipm_xyz,
            "t": self._t,
        }
        return feature_map

    def _sipm_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 500
