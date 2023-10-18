"""Unit tests for GraphDefinition."""

from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.prometheus import Prometheus
from graphnet.data.constants import FEATURES

import numpy as np
from copy import deepcopy
import torch


def test_graph_definition() -> None:
    """Tests the forward pass of GraphDefinition."""
    # Test configuration
    features = FEATURES.PROMETHEUS
    perturbation_dict = {
        "sensor_pos_x": 1.4,
        "sensor_pos_y": 2.2,
        "sensor_pos_z": 3.7,
        "t": 1.2,
    }
    mock_data = np.array([[1, 5, 2, 3], [2, 9, 6, 2]])
    seed = 42
    n_reps = 5

    graph_definition = KNNGraph(
        detector=Prometheus(), perturbation_dict=perturbation_dict, seed=seed
    )
    original_output = graph_definition(
        input_features=deepcopy(mock_data), input_feature_names=features
    )

    for _ in range(n_reps):
        graph_definition_perturbed = KNNGraph(
            detector=Prometheus(), perturbation_dict=perturbation_dict
        )

        graph_definition = KNNGraph(
            detector=Prometheus(),
            perturbation_dict=perturbation_dict,
            seed=seed,
        )

        data = graph_definition(
            input_features=deepcopy(mock_data), input_feature_names=features
        )

        perturbed_data = graph_definition_perturbed(
            input_features=deepcopy(mock_data), input_feature_names=features
        )

        assert ~torch.equal(data.x, perturbed_data.x)  # should not be equal.
        assert torch.equal(data.x, original_output.x)  # should be equal.
