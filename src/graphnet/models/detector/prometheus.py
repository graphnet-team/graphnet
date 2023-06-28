"""Prometheus-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch

from graphnet.models.detector.detector import Detector


class Prometheus(Detector):
    """`Detector` class for Prometheus prototype."""

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
        return ((x / 1.05e04) - 1.0) * 20.0
