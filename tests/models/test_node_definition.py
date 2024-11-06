"""Unit tests for node definitions."""

import numpy as np
import pandas as pd
import sqlite3
import torch
from graphnet.models.graphs.nodes import PercentileClusters
from graphnet.constants import EXAMPLE_DATA_DIR


def test_percentile_cluster() -> None:
    """Test that percentiles outputted by PercentileCluster.

    Here we check that it matches percentiles obtained from "traditional" ways.
    """
    # definitions
    percentiles = [0, 10, 50, 90, 100]
    database = f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"
    #  Grab first event in database
    with sqlite3.connect(database) as con:
        query = "select event_no from mc_truth limit 1"
        event_no = pd.read_sql(query, con)
        query = f'select sensor_pos_x, sensor_pos_y, sensor_pos_z, t from total where event_no = {str(event_no["event_no"][0])}'  # noqa: E501
        df = pd.read_sql(query, con)

    # Save original feature names, create variables.
    original_features = list(df.columns)
    x = np.array(df)
    tensor = torch.tensor(x)

    # Construct node definition
    # This defines each DOM as a cluster, and will summarize pulses seen by
    # DOMs using percentiles.
    node_definition = PercentileClusters(
        cluster_on=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"],
        percentiles=percentiles,
        input_feature_names=original_features,
    )

    # Apply node definition to torch tensor with raw pulses
    graph, new_features = node_definition(tensor)
    x_tilde = graph.x.numpy()

    # Calculate percentiles "the normal way" and compare that output of
    # node definition match.

    unique_doms = (
        df.groupby(["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"])
        .size()
        .reset_index()
    )
    for i in range(len(unique_doms)):
        idx_original = (
            (df["sensor_pos_x"] == unique_doms["sensor_pos_x"][i])
            & ((df["sensor_pos_y"] == unique_doms["sensor_pos_y"][i]))
            & (df["sensor_pos_z"] == unique_doms["sensor_pos_z"][i])
        )
        idx_tilde = (
            (
                x_tilde[:, new_features.index("sensor_pos_x")]
                == unique_doms["sensor_pos_x"][i]
            )
            & (
                x_tilde[:, new_features.index("sensor_pos_y")]
                == unique_doms["sensor_pos_y"][i]
            )
            & (
                x_tilde[:, new_features.index("sensor_pos_z")]
                == unique_doms["sensor_pos_z"][i]
            )
        )
        for percentile in percentiles:
            pct_idx = new_features.index(f"t_pct{percentile}")
            try:
                assert np.isclose(
                    x_tilde[idx_tilde, pct_idx],
                    np.percentile(df.loc[idx_original, "t"], percentile),
                )
            except AssertionError as e:
                print(f"Percentile {percentile} does not match.")
                raise e
