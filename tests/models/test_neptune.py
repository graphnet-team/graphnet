"""Unit tests for the Neptune backbone and its FPS components."""

import os
from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graphnet.models import Model, StandardModel
from graphnet.models.components import fps
from graphnet.models.components.fps import (
    farthest_point_sampling,
    farthest_point_sampling_with_assign,
    farthest_point_sampling_with_knn,
    nearest_assign,
)
from graphnet.models.components.tokenizers import FPSTokenizer
from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.models.transformer import Neptune
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.imports import has_torch_fps_package

FEATURES = ["x", "y", "z", "t", "charge", "aux"]
N_FEATURES = len(FEATURES)
NUM_PATCHES = 8
NAMES: Dict[str, Any] = dict(
    input_feature_names=FEATURES,
    coordinate_columns=["x", "y", "z"],
    time_column="t",
)

# Small enough to keep every test in the sub-second range.
TINY: Dict[str, Any] = dict(
    num_patches=NUM_PATCHES,
    d_model=32,
    depth=1,
    num_heads=2,
    hidden_dim=24,
    tokenizer_mlp_layers=[16, 16],
)


def _model(**kwargs: Any) -> Neptune:
    """Build a tiny `Neptune` over the six-column feature layout."""
    config: Dict[str, Any] = dict(NAMES, charge_column="charge", **TINY)
    config.update(kwargs)
    return Neptune(**config)


def _batch(counts: List[int], seed: int = 0) -> Batch:
    """Build a batch of events holding `counts[i]` pulses each."""
    torch.manual_seed(seed)
    data = []
    for n in counts:
        x = torch.randn(n, N_FEATURES)
        x[:, 4] = x[:, 4].abs()  # physical charge is non-negative
        data.append(Data(x=x))
    return Batch.from_data_list(data)


def test_forward_shape() -> None:
    """Test that the backbone returns one row of `nb_outputs` per event."""
    model = _model()
    counts = [3, 5, 11]
    output = model(_batch(counts))
    assert output.shape == (len(counts), model.nb_outputs)
    assert model.nb_outputs == TINY["d_model"]
    assert model.nb_inputs == N_FEATURES
    assert torch.isfinite(output).all()


def test_output_dim_overrides_nb_outputs() -> None:
    """Test that `output_dim` sets the readout width."""
    model = _model(output_dim=3)
    assert model.nb_outputs == 3
    assert model(_batch([4, 9])).shape == (2, 3)


def test_both_tokenizer_paths() -> None:
    """Test events below, at, and above the token budget in one batch."""
    model = _model()
    counts = [NUM_PATCHES - 3, NUM_PATCHES, NUM_PATCHES * 5]
    output = model(_batch(counts))
    assert output.shape == (3, model.nb_outputs)
    assert torch.isfinite(output).all()


def test_zero_hit_event() -> None:
    """Test that empty events keep their row and do not produce NaN."""
    model = _model()
    counts = [4, 0, 7, 0]  # empty in the middle and at the end
    output = model(_batch(counts))
    assert output.shape == (len(counts), model.nb_outputs)
    assert torch.isfinite(output).all()


def test_all_events_empty() -> None:
    """Test the degenerate batch in which no event has any pulse."""
    model = _model()
    output = model(_batch([0, 0]))
    assert output.shape == (2, model.nb_outputs)
    assert torch.isfinite(output).all()


def test_gradients_reach_every_stage() -> None:
    """Test that gradients flow to tokenizer, encoder, and readout."""
    model = _model()
    model(_batch([6, NUM_PATCHES * 4])).sum().backward()
    parameters = dict(model.named_parameters())
    for name in [
        "tokenizer.mlp1.0.weight",
        "tokenizer.mlp2.0.weight",
        "tokenizer.rel_encoder.0.weight",
        "encoder.pos_mlp.0.weight",
        "encoder.layers.layers.0.qkv_proj.weight",
        "encoder.layers.layers.0.ffn.w13.weight",
        "encoder.layers.layers.0.gamma_1",
        "encoder.norm.weight",
        "head.0.weight",
        "head.3.weight",
    ]:
        assert name in parameters, f"missing parameter {name}"
        gradient = parameters[name].grad
        assert gradient is not None, f"no gradient for {name}"
        assert torch.isfinite(gradient).all(), f"NaN gradient in {name}"


def test_attention_pooling() -> None:
    """Test the attention pooling variant, including on empty events."""
    model = _model(pool_type="attention")
    output = model(_batch([0, 5, NUM_PATCHES * 3]))
    assert output.shape == (3, model.nb_outputs)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "tokenizer_kwargs",
    [
        {"assign_mode": "knn", "k_neighbors": 3},
        {"lloyd_iters": 2},
        {"knn_pool": "max_mean"},
    ],
)
def test_tokenizer_variants(tokenizer_kwargs: Dict[str, Any]) -> None:
    """Test the alternative tokenizer assignment and pooling modes."""
    model = _model(tokenizer_kwargs=tokenizer_kwargs)
    output = model(_batch([4, NUM_PATCHES * 4]))
    assert output.shape == (2, model.nb_outputs)
    assert torch.isfinite(output).all()


def test_feature_names_resolve_to_columns() -> None:
    """Test that feature names select the right columns of `data.x`."""
    model = _model(input_feature_names=["t", "aux", "x", "charge", "y", "z"])
    assert model._coordinate_index == [2, 4, 5]
    assert model._time_index == [0]
    assert model._charge_index == 3
    assert model._feature_index == [1]
    assert model.tokenizer.mlp1[0].in_features == 2  # log1p(charge), aux

    # Without a charge column every pulse is a unit-weight hit.
    chargeless = Neptune(
        **dict(NAMES, input_feature_names=["x", "y", "z", "t"]), **TINY
    )
    assert chargeless.tokenizer.mlp1[0].in_features == 1
    batch = Batch.from_data_list([Data(x=torch.randn(n, 4)) for n in (3, 20)])
    assert torch.isfinite(chargeless(batch)).all()


@pytest.mark.parametrize(
    "override",
    [
        {"coordinate_columns": ["x", "y", "w"]},
        {"time_column": "time"},
        {"charge_column": "q"},
        {"feature_columns": ["aux", "missing"]},
    ],
)
def test_unknown_feature_names_raise(override: Dict[str, Any]) -> None:
    """Test that a name missing from `input_feature_names` raises."""
    with pytest.raises(ValueError, match="not in input_feature_names"):
        _model(**override)


def test_time_shift_invariance() -> None:
    """Test that a constant time offset leaves the output unchanged.

    Time only enters through differences, which is why Neptune needs no
    per-event time centering.
    """
    model = _model(tokenizer_kwargs={"lloyd_iters": 2}).eval()
    batch = _batch([5, NUM_PATCHES * 4, NUM_PATCHES * 2])
    with torch.no_grad():
        reference = model(batch)
        batch.x[:, 3] += 0.75
        shifted = model(batch)
    assert torch.allclose(shifted, reference, atol=1e-5)


def test_low_precision_without_autocast() -> None:
    """Test a model converted to low precision, as in `"bf16-true"`."""
    for dtype in (torch.bfloat16, torch.half):
        model = _model().to(dtype)
        batch = _batch([4, NUM_PATCHES * 3, 0])
        batch.x = batch.x.to(dtype)
        output = model(batch)
        assert output.shape == (3, model.nb_outputs)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def test_invalid_configurations() -> None:
    """Test that misconfigurations fail fast with a clear message."""
    with pytest.raises(ValueError, match="divisible"):
        _model(d_model=33, num_heads=2)
    with pytest.raises(ValueError, match="rotary embedding"):
        _model(num_heads=8)  # head_dim = 4, below the RoPE4D minimum of 8
    with pytest.raises(ValueError, match="3 columns"):
        _model(coordinate_columns=["x", "y"])
    with pytest.raises(ValueError, match="charge_col"):
        _model(tokenizer_kwargs={"charge_col": 1})


def test_model_config_round_trip(tmp_path: Any) -> None:
    """Test that `Neptune` survives a `ModelConfig` round trip."""
    path = os.path.join(str(tmp_path), "neptune.yml")
    model = _model(
        output_dim=5,
        pool_type="attention",
        position_encoding_schema={0: (9.0, 50.0), 1: 9.0, 3: (2.0, 10.0)},
    )
    model.save_config(path)

    loaded_config = ModelConfig.load(path)
    assert isinstance(loaded_config, ModelConfig)
    assert loaded_config == model.config

    reconstructed = Model.from_config(loaded_config)
    assert reconstructed.config == model.config
    assert repr(reconstructed) == repr(model)
    assert set(reconstructed.state_dict()) == set(model.state_dict())


def test_standard_model_end_to_end() -> None:
    """Test `Neptune` as the backbone of a `StandardModel`."""
    data_representation = EdgelessGraph(
        detector=Prometheus(), node_definition=NodesAsPulses()
    )
    names = data_representation.output_feature_names
    backbone = Neptune(
        input_feature_names=names,
        coordinate_columns=names[:3],
        time_column=names[3],
        **TINY,
    )
    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels="total_energy",
        loss_function=LogCoshLoss(),
    )
    model = StandardModel(
        data_representation=data_representation,
        backbone=backbone,
        tasks=[task],
    )
    torch.manual_seed(0)
    batch = Batch.from_data_list(
        [Data(x=torch.randn(n, len(names))) for n in (4, 20, 0)]
    )
    predictions = model(batch)
    assert len(predictions) == 1
    assert predictions[0].shape == (3, 1)


def test_runs_without_torch_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the pure-PyTorch FPS path used when `torch_fps` is missing."""
    monkeypatch.setattr(fps, "has_torch_fps_package", lambda: False)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        assert fps._torch_fps() is None
        model = _model(tokenizer_kwargs={"lloyd_iters": 1}).to(device)
        output = model(_batch([5, NUM_PATCHES * 4]).to(device))
        assert torch.isfinite(output).all()


@pytest.mark.skipif(not has_torch_fps_package(), reason="needs torch_fps")
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_torch_fps_matches_reference(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Test that the `torch_fps` backends agree with the reference."""
    torch.manual_seed(0)
    points = torch.randn(4, 300, 4, device=device)
    mask = torch.rand(4, 300, device=device) < 0.9
    start = torch.zeros(4, dtype=torch.long, device=device)

    def run() -> List[Tensor]:
        idx, assign = farthest_point_sampling_with_assign(
            points, mask, 16, start_idx=start
        )
        knn = farthest_point_sampling_with_knn(
            points, mask, 16, 8, start_idx=start
        )[1]
        return [idx, torch.where(mask, assign, 0), knn]

    fast = run()
    monkeypatch.setattr(fps, "has_torch_fps_package", lambda: False)
    for a, b in zip(fast, run()):
        assert torch.equal(a, b)


# ------------------------------------------------------------------------
# Farthest point sampling
# ------------------------------------------------------------------------


def test_fps_matches_brute_force() -> None:
    """Test FPS against an explicit greedy implementation."""
    torch.manual_seed(0)
    points = torch.randn(2, 40, 4)
    mask = torch.ones(2, 40, dtype=torch.bool)
    start = torch.zeros(2, dtype=torch.long)
    idx = farthest_point_sampling(points, mask, 6, start_idx=start)

    for b in range(2):
        chosen = [0]
        for _ in range(5):
            dist = (
                (points[b].unsqueeze(1) - points[b][chosen].unsqueeze(0))
                .square()
                .sum(-1)
                .min(dim=1)
                .values
            )
            dist[chosen] = -float("inf")
            chosen.append(int(dist.argmax()))
        assert idx[b].tolist() == chosen


def test_fps_ties_pick_lowest_index() -> None:
    """Test that exactly-tied candidates resolve to the lowest index."""
    # Four points on a line: 0 and 3 are the extremes, 1 and 2 tie.
    points = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    idx = farthest_point_sampling(
        points, mask, 4, start_idx=torch.zeros(1, dtype=torch.long)
    )
    # After 0 and 3, points 1 and 2 are both at distance 1; 1 wins.
    assert idx[0].tolist() == [0, 3, 1, 2]


def test_fps_masked_and_exhausted_rows() -> None:
    """Test that invalid points are skipped and short rows repeat."""
    points = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    mask = torch.tensor([[True, False, False, True, False, False]])
    idx = farthest_point_sampling(
        points,
        mask,
        4,
        start_idx=torch.zeros(1, dtype=torch.long),
        validate=False,
    )
    # Only indices 0 and 3 are selectable; once exhausted the last
    # selection repeats.
    assert idx[0].tolist() == [0, 3, 3, 3]


def test_fps_with_assign_partitions_points() -> None:
    """Test that Voronoi assignment sends each point to its closest
    centroid."""
    torch.manual_seed(0)
    points = torch.randn(3, 30, 4)
    mask = torch.ones(3, 30, dtype=torch.bool)
    idx, assign = farthest_point_sampling_with_assign(
        points, mask, 5, start_idx=torch.zeros(3, dtype=torch.long)
    )
    assert idx.shape == (3, 5)
    assert assign.shape == (3, 30)
    assert int(assign.min()) >= 0 and int(assign.max()) < 5
    for b in range(3):
        cents = points[b][idx[b]]
        expected = (
            (points[b].unsqueeze(1) - cents.unsqueeze(0))
            .square()
            .sum(-1)
            .argmin(dim=1)
        )
        assert torch.equal(assign[b], expected)


def test_fps_with_knn_is_sorted_and_padded() -> None:
    """Test that kNN neighbours are closest-first and short rows padded."""
    torch.manual_seed(0)
    points = torch.randn(1, 12, 4)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[0, 4:] = False  # only 4 valid points, fewer than k_neighbors
    idx, neighbors = farthest_point_sampling_with_knn(
        points,
        mask,
        3,
        6,
        start_idx=torch.zeros(1, dtype=torch.long),
        validate=False,
    )
    assert neighbors.shape == (1, 3, 6)
    # The nearest neighbour of a centroid is the centroid itself.
    assert torch.equal(neighbors[0, :, 0], idx[0])
    # Slots beyond the four valid points fall back to the centroid index.
    assert torch.equal(neighbors[0, :, 4:], idx[0].unsqueeze(-1).expand(-1, 2))


def test_nearest_assign_matches_brute_force() -> None:
    """Test segmented nearest-centroid assignment against brute force."""
    torch.manual_seed(0)
    counts = torch.tensor([7, 11, 5])
    starts = torch.cumsum(counts, 0) - counts
    points = torch.randn(int(counts.sum()), 4)
    cents = torch.randn(3 * 4, 4)
    got = nearest_assign(points, cents, starts, counts, int(counts.max()), 4)

    seg = torch.repeat_interleave(torch.arange(3), counts)
    expected = (
        (points[:, None, :] - cents.view(3, 4, 4)[seg]).square().sum(-1)
    ).argmin(1)
    assert torch.equal(got, expected)


def test_fps_rejects_bad_input() -> None:
    """Test the validation performed by the FPS front-end."""
    points = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="K <= number of valid points"):
        farthest_point_sampling(points, mask, 9)
    with pytest.raises(ValueError, match=r"\[B, N, D\]"):
        farthest_point_sampling(points[0], mask, 2)
    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        farthest_point_sampling_with_knn(points, mask, 2, 0)
