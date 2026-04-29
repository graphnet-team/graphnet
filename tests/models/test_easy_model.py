"""Unit tests for `EasySyntax.predict_as_dataframe`.

Uses a minimal `EasySyntax` subclass and a synthetic in-memory dataset so
the tests exercise only the prediction/attribute-gathering machinery,
not GNN/Detector/Task code.
"""

from typing import List, Union

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graphnet.models.easy_model import EasySyntax


class _FakeTask(torch.nn.Module):
    def __init__(self, prediction_labels: List[str]) -> None:
        super().__init__()
        self._prediction_labels = list(prediction_labels)
        self._target_labels: List[str] = []

    def inference(self) -> None:
        pass

    def train_eval(self) -> None:
        pass


class _FakeModel(EasySyntax):
    """Echoes `event_id` as the prediction so we can verify alignment."""

    def __init__(self, prediction_labels: List[str]) -> None:
        super().__init__(tasks=[_FakeTask(prediction_labels)])
        self._dummy = torch.nn.Linear(1, 1)  # so Lightning sees parameters

    def validate_tasks(self) -> None:
        pass

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Tensor]:
        if isinstance(data, list):
            data = data[0]
        # Per-event prediction: repeat event_id across all prediction columns.
        eid = data.event_id.to(torch.float32).unsqueeze(1)
        n_cols = len(self.prediction_labels)
        return [eid.repeat(1, n_cols)]

    def compute_loss(self, preds, data, verbose=False):  # type: ignore[override]
        raise NotImplementedError

    def shared_step(self, batch, batch_idx):  # type: ignore[override]
        raise NotImplementedError


def _make_dataset(n_events: int) -> List[Data]:
    """One Data per event with event_id, scalar value, pulse-level array."""
    out = []
    for i in range(n_events):
        n_pulses = 2 + (i % 3)
        out.append(
            Data(
                x=torch.zeros(n_pulses, 1),
                event_id=torch.tensor([i], dtype=torch.long),
                value=torch.tensor([float(i) * 10.0]),
                pulses=torch.arange(n_pulses, dtype=torch.float32),
                n_pulses=torch.tensor([n_pulses]),
            )
        )
    return out


def _loader(
    dataset: List[Data], batch_size: int = 4, shuffle: bool = False
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@pytest.fixture
def model() -> _FakeModel:
    return _FakeModel(prediction_labels=["pred_a", "pred_b"])


def test_predict_as_dataframe_shape_and_columns(model: _FakeModel) -> None:
    """No attributes: columns and length match prediction labels/dataset."""
    ds = _make_dataset(10)
    df = model.predict_as_dataframe(_loader(ds, batch_size=3))
    assert list(df.columns) == ["pred_a", "pred_b"]
    assert len(df) == len(ds)
    # _FakeModel echoes event_id, so both columns should equal 0..9.
    np.testing.assert_array_equal(
        df["pred_a"].to_numpy(), np.arange(10, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        df["pred_b"].to_numpy(), np.arange(10, dtype=np.float32)
    )


def test_predict_as_dataframe_attributes_aligned(model: _FakeModel) -> None:
    """Additional attributes line up with predictions row-for-row."""
    ds = _make_dataset(8)
    df = model.predict_as_dataframe(
        _loader(ds, batch_size=3),
        additional_attributes=["event_id", "value"],
    )
    assert list(df.columns) == ["pred_a", "pred_b", "event_id", "value"]
    np.testing.assert_array_equal(
        df["event_id"].to_numpy(), np.arange(8)
    )
    np.testing.assert_array_equal(
        df["value"].to_numpy(), np.arange(8, dtype=np.float32) * 10.0
    )
    # Prediction (= event_id) must equal the event_id attribute.
    np.testing.assert_array_equal(
        df["pred_a"].to_numpy().astype(np.int64),
        df["event_id"].to_numpy(),
    )


class _CountingLoader(DataLoader):
    """`DataLoader` that records how many times `__iter__` was called.

    Python's ``iter(obj)`` dispatches via ``type(obj).__iter__``, so the
    counter has to live on the class — patching the instance has no effect.
    """

    def __iter__(self):  # type: ignore[override]
        type(self).iter_count = getattr(type(self), "iter_count", 0) + 1
        return super().__iter__()


def test_predict_as_dataframe_iterates_dataloader_once(
    model: _FakeModel,
) -> None:
    """Regression: with attributes, the dataloader is iterated exactly once.

    The previous implementation iterated a second time after `predict()`
    purely to collect `additional_attributes`.
    """
    ds = _make_dataset(6)
    loader = _CountingLoader(ds, batch_size=2)
    _CountingLoader.iter_count = 0

    model.predict_as_dataframe(
        loader, additional_attributes=["event_id", "value"]
    )
    assert _CountingLoader.iter_count == 1


def test_predict_as_dataframe_with_shuffled_loader(
    model: _FakeModel,
) -> None:
    """Shuffling is allowed now that attributes come from the same pass."""
    torch.manual_seed(0)
    ds = _make_dataset(12)
    loader = _loader(ds, batch_size=4, shuffle=True)
    df = model.predict_as_dataframe(
        loader, additional_attributes=["event_id", "value"]
    )
    assert len(df) == 12
    # event_id may appear in any order, but pred and attr stay paired.
    np.testing.assert_array_equal(
        df["pred_a"].to_numpy().astype(np.int64),
        df["event_id"].to_numpy(),
    )
    np.testing.assert_array_equal(
        df["value"].to_numpy(),
        df["event_id"].to_numpy().astype(np.float32) * 10.0,
    )
    # All events present exactly once.
    assert sorted(df["event_id"].to_numpy().tolist()) == list(range(12))


def test_predict_as_dataframe_respects_limit_predict_batches(
    model: _FakeModel,
) -> None:
    """`limit_predict_batches` truncates predictions and attributes together."""
    ds = _make_dataset(20)
    df = model.predict_as_dataframe(
        _loader(ds, batch_size=4),
        additional_attributes=["event_id"],
        limit_predict_batches=2,
    )
    # 2 batches * 4 events = 8 rows; predictions and attrs must still align.
    assert len(df) == 8
    np.testing.assert_array_equal(
        df["pred_a"].to_numpy().astype(np.int64),
        df["event_id"].to_numpy(),
    )


def test_predict_as_dataframe_skips_misaligned_attribute(
    model: _FakeModel,
) -> None:
    """Pulse-level attr requested on event-level model is dropped, not crashed."""
    ds = _make_dataset(6)
    df = model.predict_as_dataframe(
        _loader(ds, batch_size=2),
        additional_attributes=["event_id", "pulses"],
    )
    assert "event_id" in df.columns
    assert "pulses" not in df.columns
    assert len(df) == 6
