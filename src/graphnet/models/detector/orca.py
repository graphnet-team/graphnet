"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector


class ORCA115(Detector):
    """`Detector` class for ORCA-115."""

    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_xy,
            "pos_y": self._dom_xy,
            "pos_z": self._dom_z,
            "dir_x": self._dir_xy,
            "dir_y": self._dir_xy,
            "dir_z": self._dir_z,
            "tot": self._tot,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x - 117.5) / 7.75

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1800) / 180

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # return torch.log10(x)
        return (x - 75) / 7.5

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.275) * 12.9


class ORCA6(Detector):
    """`Detector` class for ORCA-6."""

    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_xy,
            "dir_y": self._dir_xy,
            "dir_z": self._dir_z,
            "tot": self._tot,
        }
        return feature_map

    def _dom_x(self, x: torch.tensor) -> torch.tensor:
        return (x - 457.8) * 0.37

    def _dom_y(self, x: torch.tensor) -> torch.tensor:
        return (x - 574.1) * 1.04

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x - 108.6) * 0.12

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1025) * 0.021

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # return torch.log10(x)
        return (x - 117) * 0.085

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.23) * 12.9
