"""Prometheus-specific `Detector` class(es)."""

from torch_geometric.data import Data

from graphnet.models.detector.detector import Detector


class Prometheus(Detector):
    """`Detector` class for Prometheus prototype."""

    features = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"]

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
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

        return data
