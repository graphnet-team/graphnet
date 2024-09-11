"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"
    sensor_time_column = "dom_time"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeKaggle(Detector):
    """`Detector` class for Kaggle Competition."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["x", "y", "z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"
    sensor_time_column = "time"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "x": self._xyz,
            "y": self._xyz,
            "z": self._xyz,
            "time": self._time,
            "charge": self._charge,
            "auxiliary": self._identity,
        }
        return feature_map

    def _xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 3.0


class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xy,
            "dom_y": self._dom_xy,
            "dom_z": self._dom_z,
            "dom_time": self._dom_time,
            "charge": self._identity,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350.0) / 100.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return ((x / 1.05e04) - 1.0) * 20.0

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeUpgrade(Detector):
    """`Detector` class for IceCube-Upgrade."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"
    sensor_time_column = "dom_time"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._identity,
            "pmt_area": self._pmt_area,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "hlc": self._identity,
        }

        return feature_map

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x / 2e04) - 1.0

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 2.0

    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x - 50.0) / 50.0

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 20.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return (x - 60.0) / 60.0

    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 130.0

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05
