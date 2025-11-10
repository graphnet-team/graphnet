"""MAGIC-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import MAGIC_GEOMETRY_TABLE_DIR


class MAGIC(Detector):
    """`Detector` class for MAGIC telescopes."""

    geometry_table_path = os.path.join(
        MAGIC_GEOMETRY_TABLE_DIR, "magic.parquet"
    )

    # By default, treat the telescope ID as a spatial-like z-coordinate
    xyz = ["x_cam", "y_cam", "tel_id"]

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension.

        Note: tel_id can take the integer values 0 or 1, where:
            - 0 corresponds to MAGIC-I
            - 1 corresponds to MAGIC-II
        """
        feature_map = {
            "x_cam": self._xy,
            "y_cam": self._xy,
            "tel_id": self._identity,
            "time": self._time,
            "charge": self._charge,
            "telescope_phi": self._identity,
            "telescope_theta": self._identity,
        }
        return feature_map

    def _xy(self, x: torch.tensor) -> torch.tensor:
        return x / 28.5

    def _time(self, x: torch.tensor) -> torch.tensor:
        """Scale time based on the average time of arrival."""
        t_min = -30
        t_max = 60
        return (x - t_min) / (t_max - t_min)

    def _charge(self, x: torch.tensor) -> torch.tensor:
        """Add a small epsilon to avoid log(0)."""
        epsilon = 1e-6
        return torch.log10(x + epsilon)
