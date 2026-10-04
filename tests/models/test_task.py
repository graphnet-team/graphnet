"""Unit tests for Task classes."""

from typing import Any

import pytest
import torch

from graphnet.data.constants import FEATURES
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithLogKappa,
    EnergyReconstruction,
)
from graphnet.training.loss_functions import LogCoshLoss, VonMisesFisher3DLoss
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses


def test_transform_prediction_and_target() -> None:
    """Test implementation of `transform_*` arguments to `Task`."""
    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.DEEPCORE,
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


def _log_kappa_task(**kwargs: Any) -> DirectionReconstructionWithLogKappa:
    return DirectionReconstructionWithLogKappa(
        hidden_size=8,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
        **kwargs,
    )


def test_log_kappa_direction_output() -> None:
    """Unit directions, kappa above its floor, the stock output layout."""
    task = _log_kappa_task(kappa_min=1.0)
    assert task.nb_inputs == 4
    x = torch.randn(64, 4)
    pred = task._forward(x)
    assert pred.shape == (64, 4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(pred[:, :3], dim=1), torch.ones(64)
    )
    torch.testing.assert_close(pred[:, 3], 1.0 + torch.exp(x[:, 3]))


def test_log_kappa_no_ceiling_and_no_overflow() -> None:
    """Kappa grows past the stock head's typical ceiling and stays finite."""
    task = _log_kappa_task()
    x = torch.tensor([[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 1e4]])
    kappa = task._forward(x)[:, 3]
    assert kappa[0] > 2e4
    assert torch.isfinite(kappa).all()


def test_log_kappa_floor_must_be_positive() -> None:
    """A zero floor lets training collapse, so it is rejected."""
    with pytest.raises(ValueError):
        _log_kappa_task(kappa_min=0.0)


def test_log_kappa_loss_and_gradients() -> None:
    """The vMF loss is finite and its gradient reaches all four outputs."""
    task = _log_kappa_task()
    x = torch.randn(32, 4, requires_grad=True)
    target = torch.nn.functional.normalize(torch.randn(32, 3), dim=1)
    loss = task._loss_function(task._forward(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert (x.grad.abs().sum(dim=0) > 0).all()
