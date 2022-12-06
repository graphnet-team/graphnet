"""Unit tests for Task classes."""

import pytest
import torch

from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss


def test_transform_prediction_and_target() -> None:
    """Test implementation of `transform_*` arguments to `Task`."""
    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
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
