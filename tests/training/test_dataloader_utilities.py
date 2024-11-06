"""Unit tests for dataloader utilities.

@NOTE: These utility methods should be deprecated in favour of the indicated
member methods in `DataLoader`.
"""

from typing import Tuple

import pytest

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.constants import TEST_PARQUET_DATA, TEST_SQLITE_DATA
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.training.utils import make_train_validation_dataloader

# Configuration
NB_EVENTS_TOTAL = 5

graph_definition = KNNGraph(
    detector=IceCubeDeepCore(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
    input_feature_names=FEATURES.DEEPCORE,
)


# Unit test(s)
def test_none_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    (
        train_dataloader,
        test_dataloader,
    ) = make_train_validation_dataloader(
        db=TEST_SQLITE_DATA,
        graph_definition=graph_definition,
        selection=None,
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == NB_EVENTS_TOTAL


@pytest.mark.parametrize(
    "selection",
    [
        (0, 1, 2, 3, 4),
        (0, 1, 3, 4),
        (0, 1),
    ],
)
def test_array_selection(selection: Tuple[int]) -> None:
    """Test agreement of the two ways to calculate this loss."""
    (train_dataloader, test_dataloader) = make_train_validation_dataloader(
        db=TEST_SQLITE_DATA,
        graph_definition=graph_definition,
        selection=list(selection),
        pulsemaps=["SRTInIcePulses"],
        features=FEATURES.DEEPCORE,
        truth=TRUTH.DEEPCORE,
        batch_size=1,
    )

    assert len(train_dataloader) + len(test_dataloader) == len(selection)


def test_empty_selection() -> None:
    """Test agreement of the two ways to calculate this loss."""
    try:
        _ = make_train_validation_dataloader(
            db=TEST_SQLITE_DATA,
            graph_definition=graph_definition,
            selection=list(),
            pulsemaps=["SRTInIcePulses"],
            features=FEATURES.DEEPCORE,
            truth=TRUTH.DEEPCORE,
            batch_size=1,
        )
        assert False  # Is expected to throw `ValueError`.
    except ValueError:
        pass


def test_parquet() -> None:
    """Test agreement of the two ways to calculate this loss."""
    try:
        _ = make_train_validation_dataloader(
            db=TEST_PARQUET_DATA,
            graph_definition=graph_definition,
            selection=None,
            pulsemaps=["SRTInIcePulses"],
            features=FEATURES.DEEPCORE,
            truth=TRUTH.DEEPCORE,
            batch_size=1,
        )
        assert False  # Is expected to throw `AssertionError`.
    except AssertionError as e:
        assert str(e).startswith("Format of input file")
        assert str(e).endswith("is not supported.")
