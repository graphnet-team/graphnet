"""Unit tests for Task classes."""

import pytest
import torch

from graphnet.data.constants import FEATURES
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses


def test_transform_prediction_and_target() -> None:
    """Test implementation of `transform_*` arguments to `Task`."""
    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        node_feature_names=FEATURES.DEEPCORE,
    )
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
    )

    # Test not inverse functions
    with pytest.raises(
        AssertionError,
        match=(
            "The provided transforms for targets during training and "
            "predictions during inference are not inverse. Please adjust "
            "transformation functions or support."
        ),
    ):
        EnergyReconstruction(
            hidden_size=gnn.nb_outputs,
            target_labels="energy",
            loss_function=LogCoshLoss(),
            transform_target=torch.log10,
            transform_inference=torch.log10,
        )
