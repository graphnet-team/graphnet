"""Unit tests for ModelConfig class."""

import os.path

import pytest
import torch
from torch.optim.adam import Adam

from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss


def test_simple_model_config(path: str = "/tmp/simple_model.yml") -> None:
    """Test saving, loading, and reconstructing simple model."""
    # Construct single Model
    model = DynEdge(
        nb_inputs=9,
        global_pooling_schemes=["min", "max", "mean", "sum"],
        add_global_variables_after_pooling=True,
    )

    # Save config to file
    model.save_config(path.replace(".yml", ""))
    assert os.path.exists(path)
    model.save_config(path)

    # Load config from file
    loaded_config = ModelConfig.load(path)
    assert isinstance(loaded_config, ModelConfig)
    assert loaded_config == model.config

    # Construct model
    constructed_model = Model.from_config(loaded_config)
    assert constructed_model.config == model.config
    assert repr(constructed_model) == repr(model)


def test_nested_model_config(path: str = "/tmp/nested_model.yml") -> None:
    """Test saving, loading, and reconstructing nested model."""
    # Construct nested Model
    model = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )

    # Save config to file
    model.save_config(path)
    assert os.path.exists(path)

    # Load config from file
    loaded_config = ModelConfig.load(path)
    assert isinstance(loaded_config, ModelConfig)
    assert loaded_config == model.config

    # Construct model
    constructed_model = Model.from_config(loaded_config)
    assert constructed_model.config == model.config
    assert repr(constructed_model) == repr(model)


def test_complete_model_config(path: str = "/tmp/complete_model.yml") -> None:
    """Test saving, loading, and reconstructing nested model."""
    # Construct StandardModel
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels="energy",
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
    )

    # Save config to file
    model.save_config(path)
    assert os.path.exists(path)

    # Load config from file
    loaded_config = ModelConfig.load(path)
    assert isinstance(loaded_config, ModelConfig)
    # NB: Using _as_dict rather than straight comparison since loaded lambda
    #     functions are serialised as strings and do not match the original
    #     ones. Casting both to dict means that lambda functions are serialised
    #     to strings for both and are thus comparable.
    assert loaded_config.as_dict() == model.config.as_dict()

    # Construct model
    try:
        constructed_model = Model.from_config(loaded_config, load_modules=[])
    except ValueError:
        # Expected behaviour for Model that utilises lambda functions and non-
        # graphnet classes
        assert True
    finally:
        constructed_model = Model.from_config(
            loaded_config, trust=True, load_modules=["torch"]
        )
        assert True

    for key in [
        "coarsening",
        "optimizer_class",
        "optimizer_kwargs",
        "scheduler_class",
        "scheduler_kwargs",
        "scheduler_config",
        "detector",
        "gnn",
    ]:
        assert (
            constructed_model.config.arguments[key]
            == model.config.arguments[key]
        )
    assert len(constructed_model.config.arguments["tasks"]) == len(
        model.config.arguments["tasks"]
    )
    assert len(constructed_model.config.arguments["tasks"]) == 1
    # NB: See comment above for handling of lambda functions.
    x_ = torch.logspace(1, 5, 10)
    assert torch.all(
        constructed_model.config.arguments["tasks"][0].arguments[
            "transform_prediction_and_target"
        ](x_)
        == model.config.arguments["tasks"][0].arguments[
            "transform_prediction_and_target"
        ](x_)
    )

    assert repr(constructed_model) == repr(model)


@pytest.mark.run(after="test_complete_model_config")
def test_loading_and_saving_model_config(
    load_path: str = "/tmp/complete_model.yml",
    save_path: str = "/tmp/complete_model_duplicate.yml",
) -> None:
    """Test loading and then saving model config."""
    # Load and construct model
    loaded_config = ModelConfig.load(load_path)
    model = Model.from_config(loaded_config, trust=True)

    # Save config to file
    model.save_config(save_path)

    # Check that the configs agree
    saved_config = ModelConfig.load(save_path)
    assert saved_config == loaded_config
