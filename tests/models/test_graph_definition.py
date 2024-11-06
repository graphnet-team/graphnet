"""Unit tests for GraphDefinition."""

from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.prometheus import ORCA150SuperDense
from graphnet.data.constants import FEATURES
from graphnet.models.detector import IceCube86, IceCubeUpgrade
from graphnet.models.graphs.nodes import PercentileClusters
from graphnet.models.graphs import GraphDefinition
from graphnet.constants import EXAMPLE_DATA_DIR, TEST_DATA_DIR

import numpy as np
from copy import deepcopy
import torch
import pandas as pd
import os
import sqlite3
from typing import List


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
        detector=ORCA150SuperDense(),
        perturbation_dict=perturbation_dict,
        seed=seed,
    )
    original_output = graph_definition(
        input_features=deepcopy(mock_data), input_feature_names=features
    )

    for _ in range(n_reps):
        graph_definition_perturbed = KNNGraph(
            detector=ORCA150SuperDense(), perturbation_dict=perturbation_dict
        )

        graph_definition = KNNGraph(
            detector=ORCA150SuperDense(),
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


def get_event(
    database: str, pulsemap: str, features: List[str], truth_table: str
) -> np.ndarray:
    """Grab first event in database."""
    query_features = ", ".join(features)
    print(query_features)
    with sqlite3.connect(database) as con:
        query = f"SELECT event_no FROM {truth_table} limit 1"
        event_no = pd.read_sql(query, con)
        query = f'SELECT {query_features} FROM {pulsemap} WHERE event_no = {str(event_no["event_no"][0])}'  # noqa: E501
        df = pd.read_sql(query, con)
    return np.array(df)


def test_geometry_tables() -> None:
    """Test parts of GraphDefinition that is related to geometry tables."""
    # Test config
    databases = {
        "ORCA150SuperDense": os.path.join(
            EXAMPLE_DATA_DIR, "sqlite/prometheus/prometheus-events.db"
        ),
        "IceCube86": os.path.join(
            TEST_DATA_DIR,
            "sqlite/oscNext_genie_level7_v02/oscNext_genie_level7_v02_first_5_frames.db",  # noqa: E501
        ),
        "IceCubeUpgrade": os.path.join(
            TEST_DATA_DIR,
            "sqlite/upgrade_genie_step4_140028_000998_first_5_frames/upgrade_genie_step4_140028_000998_first_5_frames.db",  # noqa: E501
        ),
    }
    meta = {
        "ORCA150SuperDense": {"pulsemap": "total", "truth_table": "mc_truth"},
        "IceCube86": {"pulsemap": "SRTInIcePulses", "truth_table": "truth"},
        "IceCubeUpgrade": {
            "pulsemap": "SplitInIcePulses",
            "truth_table": "truth",
        },
    }

    string_mask = np.arange(0, 50, 1).tolist()

    # Tests
    for detector in [ORCA150SuperDense(), IceCube86(), IceCubeUpgrade()]:
        # Get configs for test
        database = databases[detector.__class__.__name__]
        truth_table = meta[detector.__class__.__name__]["truth_table"]
        pulsemap = meta[detector.__class__.__name__]["pulsemap"]
        feature_names = list(detector.feature_map().keys())

        # Query sqlite database for a single event
        x = get_event(database, pulsemap, feature_names, truth_table)
        # Define a node definition
        node_definition = PercentileClusters(
            cluster_on=detector.sensor_position_names,
            percentiles=[0, 50, 100],
            input_feature_names=feature_names,
        )

        # "Normal" Graph Definition
        gd_original = GraphDefinition(
            detector=detector, node_definition=node_definition
        )
        # GraphDefinition with inactive sensors
        gd_with_inactive_sensors = GraphDefinition(
            detector=detector,
            node_definition=node_definition,
            add_inactive_sensors=True,
        )

        # GraphDefinition with masked sensors
        gd_string_mask = GraphDefinition(
            detector=detector,
            node_definition=node_definition,
            add_inactive_sensors=True,
            string_mask=string_mask,
        )

        # GraphDefinition where output is sorted according to y-coordinate
        gd_sorted = graph_with_inactive_sensors = GraphDefinition(
            detector=detector,
            node_definition=node_definition,
            sort_by=detector.sensor_position_names[1],
        )

        graph_original = gd_original(deepcopy(x), feature_names)
        graph_with_inactive_sensors = gd_with_inactive_sensors(
            deepcopy(x), feature_names
        )
        graph_masked = gd_string_mask(deepcopy(x), feature_names)
        graph_sorted = gd_sorted(deepcopy(x), feature_names)

        # check dimensions
        assert (
            graph_original.x.shape[0] < graph_with_inactive_sensors.x.shape[0]
        )
        assert graph_masked.x.shape[0] < graph_with_inactive_sensors.x.shape[0]
        assert graph_masked.x.shape[0] > graph_original.x.shape[0]
        assert graph_sorted.x.shape[0] == graph_original.x.shape[0]

        # Soft check of sorting. If the order is identical, test fails.
        indices = []
        trivial_index = np.arange(0, len(graph_original.x))
        for k in range(len(graph_original.x)):
            slice = graph_original.x[k, :]
            # xyz comparison
            idx = (
                (graph_sorted.x[:, 0] == slice[0])
                & (graph_sorted.x[:, 1] == slice[1])
                & (graph_sorted.x[:, 2] == slice[2])
            )
            indices.append(trivial_index[idx])
        match = 0
        for index in range(len(trivial_index)):
            match += indices[index] == trivial_index[index]
        assert match < len(trivial_index)
