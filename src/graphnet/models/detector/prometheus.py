"""Prometheus-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import PROMETHEUS_GEOMETRY_TABLE_DIR


class ORCA150SuperDense(Detector):
    """`Detector` class for Prometheus ORCA150SuperDense."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca_150.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350) / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class TRIDENT1211(Detector):
    """`Detector` class for Prometheus TRIDENT1211."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "trident.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 1900

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeUpgrade7(Detector):
    """`Detector` class for Prometheus IceCubeUpgrade7."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class WaterDemo81(Detector):
    """`Detector` class for Prometheus WaterDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_water.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class BaikalGVD8(Detector):
    """`Detector` class for Prometheus BaikalGVD8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "gvd.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceDemo81(Detector):
    """`Detector` class for Prometheus IceDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_ice.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ARCA115(Detector):
    """`Detector` class for Prometheus ARCA115."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "arca.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ORCA150(Detector):
    """`Detector` class for Prometheus ORCA150."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCube86Prometheus(Detector):
    """`Detector` class for Prometheus IceCube86."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeDeepCore8(Detector):
    """`Detector` class for Prometheus IceCubeDeepCore8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_deepcore.parquet"
    )
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
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeGen2(Detector):
    """`Detector` class for Prometheus IceCubeGen2."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_gen2.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"
    sensor_time_column = "t"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class PONETriangle(Detector):
    """`Detector` class for Prometheus PONE Triangle."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "pone_triangle.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"
    sensor_time_column = "t"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class Prometheus(ORCA150SuperDense):
    """Reference to ORCA150SuperDense."""
