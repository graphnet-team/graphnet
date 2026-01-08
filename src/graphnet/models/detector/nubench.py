"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import NUBENCH_GEOMETRY_TABLE_DIR


class NuBenchDetector(Detector):
    """Generic Detector Class from the NuBench Paper (arXiv:2511.13111)."""

    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"
    sensor_time_column = "t"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
            "charge": self._charge,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 10e5

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(1 + x)


class FlowerS(NuBenchDetector):
    """`Detector` class for Flower S."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_s.parquet"
    )


class FlowerL(NuBenchDetector):
    """`Detector` class for Flower L."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_l.parquet"
    )


class FlowerXL(NuBenchDetector):
    """`Detector` class for Flower XL."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_xl.parquet"
    )


class Triangle(NuBenchDetector):
    """`Detector` class for Triangle."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "triangle.parquet"
    )


class Cluster(NuBenchDetector):
    """`Detector` class for Cluster."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "cluster.parquet"
    )


class Hexagon(NuBenchDetector):
    """`Detector` class for Hexagon."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "hexagon.parquet"
    )
