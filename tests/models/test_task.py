import sys
import types
from inspect import getmembers

import pytest
import torch

from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import (
    EnergyReconstruction,
    PointingReconstructionWithKappa,
)
from graphnet.training.loss_functions import LogCoshLoss, GaussianNLLLoss


def test_transform_prediction_and_target():

    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )

    # Test not inverse functions
    with pytest.raises(
        AssertionError,
        match="The provided transforms for targets during training and predictions during inference are not inverse. Please adjust transformation functions or support.",
    ):
        EnergyReconstruction(
            hidden_size=gnn.nb_outputs,
            target_labels="energy",
            loss_function=LogCoshLoss(),
            transform_target=torch.log10,
            transform_inference=torch.log10,
        )

    # Test wrong combination of inputs
    with pytest.raises(
        AssertionError,
        match="Please specify both `transform_inference` and `transform_target`",
    ):
        EnergyReconstruction(
            hidden_size=gnn.nb_outputs,
            target_labels="energy",
            loss_function=LogCoshLoss(),
            transform_prediction_and_target=torch.log10,
            transform_inference=torch.log10,
        )


def test_PointingReconstructionWithKappa():

    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )

    PointingReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=["zenith", "azimuth"],
        loss_function=GaussianNLLLoss(),
    )


def is_function_local(object):
    return isinstance(object, types.FunctionType) and object.__module__ == __name__


def main():
    # Test tasks in script
    function_names = [name for name, _ in getmembers(sys.modules[__name__], predicate=is_function_local)]
    for function in function_names[2:]:
        eval(function)


if __name__ == "__main__":
    main()
