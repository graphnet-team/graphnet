import torch
from torch_geometric.data import Data

from graphnet.models.components.pool import (
    group_pulses_to_dom,
    group_pulses_to_pmt,
    sum_pool_and_distribute,
)
from graphnet.data.constants import FEATURES
from graphnet.models.detector.detector import Detector


class Prototype(Detector):
    """`Detector` class for IceCube-86."""

    features = ["x", "y", "z", "t"]

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
        # data.x[:, 4] /= 1.0  # charge
        # data.x[:, 5] -= 1.25  # rde
        # data.x[:, 5] /= 0.25
        # data.x[:, 6] /= 0.05  # pmt_area

        return data
