"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
from torch_geometric.data import Data

from graphnet.models.components.pool import (
    group_pulses_to_dom,
    group_pulses_to_pmt,
    sum_pool_and_distribute,
)
from graphnet.data.constants import FEATURES
from graphnet.models.detector.detector import Detector


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

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
            "charge": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350.0) / 100.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return ((x / 1.05e04) - 1.0) * 20.0

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return x / 1.0


class IceCubeUpgrade(IceCube86):
    """`Detector` class for IceCube-Upgrade."""

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "pmt_area": self._pmt_area,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
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


class IceCubeUpgrade_V2(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return self.nb_inputs + 3

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Assign pulse cluster indices to DOMs and PMTs, respectively
        data = group_pulses_to_dom(data)
        data = group_pulses_to_pmt(data)

        # Feature engineering inspired by Linea Hedemark and Tetiana Kozynets.
        xyz = torch.stack((data["dom_x"], data["dom_y"], data["dom_z"]), dim=1)
        pmt_dir = torch.stack(
            (data["pmt_dir_x"], data["pmt_dir_x"], data["pmt_dir_x"]), dim=1
        )
        charge = data["charge"].unsqueeze(dim=1)
        center_of_gravity = sum_pool_and_distribute(
            xyz * charge, data.batch
        ) / sum_pool_and_distribute(charge, data.batch)
        vector_to_center_of_gravity = center_of_gravity - xyz
        distance_to_center_of_gravity = torch.norm(
            vector_to_center_of_gravity, p=2, dim=1
        )
        unit_vector_to_center_of_gravity = vector_to_center_of_gravity / (
            distance_to_center_of_gravity.unsqueeze(dim=1) + 1e-3
        )
        cos_angle_wrt_center_of_gravity = (
            pmt_dir * unit_vector_to_center_of_gravity
        ).sum(dim=1)
        photoelectrons_on_pmt = (
            sum_pool_and_distribute(data["charge"], data.pmt_index, data.batch)
            .floor()
            .clip(1, None)
        )

        # Add new features
        data.x = torch.cat(
            (
                data.x,
                photoelectrons_on_pmt.unsqueeze(dim=1),
                distance_to_center_of_gravity.unsqueeze(dim=1),
                cos_angle_wrt_center_of_gravity.unsqueeze(dim=1),
            ),
            dim=1,
        )

        # Preprocessing
        data.x[:, 0] /= 500.0  # dom_x
        data.x[:, 1] /= 500.0  # dom_y
        data.x[:, 2] /= 500.0  # dom_z
        data.x[:, 3] /= 2e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 2.0  # charge
        # data.x[:,5] /= 1.  # rde
        data.x[:, 6] /= 0.05  # pmt_area
        data.x[:, 7] -= 50.0  # string
        data.x[:, 7] /= 50.0
        data.x[:, 8] /= 20.0  # pmt_number
        data.x[:, 9] -= 60.0  # dom_number
        data.x[:, 9] /= 60.0
        # data.x[:,10] /= 1.  # pmt_dir_x
        # data.x[:,11] /= 1.  # pmt_dir_y
        # data.x[:,12] /= 1.  # pmt_dir_z
        data.x[:, 13] /= 130.0  # dom_type

        # -- Engineered features
        data.x[:, 14] = (
            torch.log10(data.x[:, 14]) / 2.0
        )  # photoelectrons_on_pmt
        data.x[:, 15] = (
            torch.log10(1e-03 + data.x[:, 15]) / 2.0
        )  # distance_to_center_of_gravity

        return data
