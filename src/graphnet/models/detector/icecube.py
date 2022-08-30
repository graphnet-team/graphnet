import torch
from torch_geometric.data import Data

from graphnet.components.pool import (
    group_pulses_to_dom,
    group_pulses_to_pmt,
    sum_pool_and_distribute,
)
from graphnet.data.constants import FEATURES
from graphnet.models.detector.detector import Detector


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    # Implementing abstract class attribute
    features = FEATURES.ICECUBE86

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 100.0  # dom_x
        data.x[:, 1] /= 100.0  # dom_y
        data.x[:, 2] += 350.0  # dom_z
        data.x[:, 2] /= 100.0
        data.x[:, 3] /= 1.05e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 3] *= 20.0
        data.x[:, 4] /= 1.0  # charge
        data.x[:, 5] -= 1.25  # rde
        data.x[:, 5] /= 0.25
        data.x[:, 6] /= 0.05  # pmt_area

        return data


class IceCubeAddedFeatures(IceCube86):
    """More features from summary statistics"""
    features = FEATURES.ICECUBE86 +\
        [
            "min_pool_time",
            "max_pool_time",
            "std_pool_time",
            "min_pool_charge",
            "max_pool_charge",
            "std_pool_charge",
            "pulses_per_dom"
        ]

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 100.0  # dom_x
        data.x[:, 1] /= 100.0  # dom_y
        data.x[:, 2] += 350.0  # dom_z
        data.x[:, 2] /= 100.0
        data.x[:, 3] /= 1.05e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 3] *= 20.0
        data.x[:, 4] /= 1.0  # charge
        data.x[:, 5] -= 1.25  # rde
        data.x[:, 5] /= 0.25
        data.x[:, 6] /= 0.05  # pmt_area
        data.x[:, 7] -= 1.3e4
        data.x[:, 7] /= 5e3
        data.x[:, 8] -= 1.4e4
        data.x[:, 8] /= 5e3
        data.x[:, 9] -= 300
        data.x[:, 9] /= 300
        data.x[:, 10] -= 0.6
        data.x[:, 10] /= 0.5
        data.x[:, 11] -= 6
        data.x[:, 11] /= 30
        data.x[:, 12] -= 1
        data.x[:, 12] /= 6
        data.x[:, 13] -= 15
        data.x[:, 13] /= 28

        return data

class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""


class IceCubeUpgrade(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Check(s)
        self._validate_features(data)

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

        return data


class IceCubeUpgrade_V2(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    @property
    def nb_outputs(self):
        return self.nb_inputs + 3

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
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
