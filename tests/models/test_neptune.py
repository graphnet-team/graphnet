"""Unit tests for the Neptune backbone and its FPS components."""

import os
from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graphnet.models import Model, StandardModel
from graphnet.models.components.fps import (
    farthest_point_sampling,
    farthest_point_sampling_with_assign,
    farthest_point_sampling_with_knn,
    nearest_assign,
)
from graphnet.models.components.tokenizers import FPSTokenizer
from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector.icecube import IceCube86, IceCubeKaggle
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.models.transformer import Neptune
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.config import ModelConfig

# Feature layout used throughout: [x, y, z, t, charge, aux].
N_FEATURES = 6
NUM_PATCHES = 8

# Small enough to keep every test in the sub-second range.
TINY: Dict[str, Any] = dict(
    num_patches=NUM_PATCHES,
    token_dim=32,
    num_layers=1,
    num_heads=2,
    hidden_dim=24,
    tokenizer_kwargs={"mlp_layers": [16, 16]},
)


def _model(**kwargs: Any) -> Neptune:
    """Build a tiny `Neptune` over the six-column feature layout."""
    config: Dict[str, Any] = dict(TINY)
    config.update(kwargs)
    return Neptune(nb_inputs=N_FEATURES, charge_column=4, **config)


def _batch(counts: List[int], seed: int = 0) -> Batch:
    """Build a batch of events holding `counts[i]` pulses each."""
    torch.manual_seed(seed)
    return Batch.from_data_list(
        [Data(x=torch.randn(n, N_FEATURES)) for n in counts]
    )


def test_forward_shape() -> None:
    """Test that the backbone returns one row of `nb_outputs` per event."""
    model = _model()
    counts = [3, 5, 11]
    output = model(_batch(counts))
    assert output.shape == (len(counts), model.nb_outputs)
    assert model.nb_outputs == TINY["token_dim"]
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
    # Fewer than, exactly, and more than `num_patches` pulses.
    counts = [NUM_PATCHES - 3, NUM_PATCHES, NUM_PATCHES * 5]
    output = model(_batch(counts))
    assert output.shape == (3, model.nb_outputs)
    assert torch.isfinite(output).all()


def test_zero_hit_event() -> None:
    """Test that empty events keep their row and do not produce NaN.

    Events with no pulses contribute no rows to `data.batch`, so a backbone
    that infers the batch size from `batch.max()` would silently drop
    trailing empty events and misalign predictions against labels.
    """
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


def test_single_event_without_batch_attribute() -> None:
    """Test that an uncollated `Data` object is treated as one event."""
    model = _model()
    torch.manual_seed(0)
    output = model(Data(x=torch.randn(12, N_FEATURES)))
    assert output.shape == (1, model.nb_outputs)


def test_gradients_reach_every_stage() -> None:
    """Test that gradients flow to tokenizer, encoder, and readout.

    The parameter names asserted here are also the `state_dict` keys of the
    reference implementation, so this doubles as a check that the module
    nesting -- and hence checkpoint compatibility -- is preserved.
    """
    model = _model()
    model(_batch([6, NUM_PATCHES * 4])).sum().backward()
    parameters = dict(model.named_parameters())
    for name in [
        "tokenizer.mlp1.0.weight",
        "tokenizer.mlp2.0.weight",
        "tokenizer.rel_encoder.0.weight",
        "encoder.abs_pos_encoder.mlp.0.weight",
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
        {"mlp_layers": [16, 16], "assign_mode": "knn", "k_neighbors": 3},
        {"mlp_layers": [16, 16], "lloyd_iters": 2},
        {"mlp_layers": [16, 16], "knn_pool": "max_mean"},
    ],
)
def test_tokenizer_variants(tokenizer_kwargs: Dict[str, Any]) -> None:
    """Test the alternative tokenizer assignment and pooling modes."""
    model = _model(tokenizer_kwargs=tokenizer_kwargs)
    output = model(_batch([4, NUM_PATCHES * 4]))
    assert output.shape == (2, model.nb_outputs)
    assert torch.isfinite(output).all()


def test_charge_scaling_and_columns() -> None:
    """Test the detector-unit adapter in front of the tokenizer."""
    # `log10` charge is converted to the `log1p` the tokenizer expects, and
    # placed in column 0 of the feature tensor.
    model = _model(charge_scaling="log10")
    x = torch.zeros(3, N_FEATURES)
    x[:, 4] = torch.log10(torch.tensor([1.0, 10.0, 100.0]))
    _, features, _, q_phys, _ = model._prepare_inputs(Data(x=x))
    assert torch.allclose(q_phys, torch.tensor([1.0, 10.0, 100.0]), atol=1e-5)
    assert torch.allclose(
        features[:, 0], torch.log1p(torch.tensor([1.0, 10.0, 100.0]))
    )
    # The remaining feature columns follow charge, in order.
    assert features.shape == (3, 1 + 1)  # charge + the single `aux` column
    assert model.tokenizer.mlp1[0].in_features == 2

    # Without a charge column every pulse counts once.
    chargeless = Neptune(nb_inputs=4, charge_column=None, **TINY)
    assert chargeless.tokenizer.mlp1[0].in_features == 1
    _, features, _, q_phys, _ = chargeless._prepare_inputs(
        Data(x=torch.randn(5, 4))
    )
    assert torch.equal(q_phys, torch.ones(5))
    assert features.shape == (5, 1)


def test_charge_scale_undoes_a_rescaled_logarithm() -> None:
    """Test `charge_scale` for detectors that rescale the log charge.

    `IceCubeKaggle` stores `log10(charge) / 3`, so undoing the scaling needs
    the column multiplied by three first.
    """
    charge = torch.tensor([0.5, 1.0, 7.3, 120.0])
    for detector, scale in ((IceCube86(), 1.0), (IceCubeKaggle(), 3.0)):
        model = _model(charge_scaling="log10", charge_scale=scale)
        x = torch.zeros(4, N_FEATURES)
        x[:, 4] = detector._charge(charge)
        _, _, _, q_phys, _ = model._prepare_inputs(Data(x=x))
        assert torch.allclose(
            q_phys, charge, rtol=1e-4
        ), f"{type(detector).__name__} charge not recovered"


def test_knn_padding_is_not_counted_as_neighbours() -> None:
    """Test that repeated-centroid padding is excluded from kNN statistics.

    When an event holds fewer points than `k_neighbors`, the sampling
    backend pads the neighbour list by repeating the centroid index. Since
    that index is itself valid, the padded slots must be masked out by rank,
    or the centroid is counted several times in the per-token multiplicity,
    total charge, mean, and spread. Asking for more neighbours than an event
    has points must therefore give the same tokens as asking for exactly as
    many as it has.
    """
    base: Dict[str, Any] = dict(
        feature_dim=2,
        max_tokens=4,
        token_dim=8,
        mlp_layers=[8],
        assign_mode="knn",
    )
    # Event 0 holds 5 points, above `max_tokens` so it takes the sampling
    # path, and below the 10 neighbours requested so its list gets padded.
    counts = [5, 40]
    torch.manual_seed(0)
    n_points = sum(counts)
    coords = torch.randn(n_points, 3)
    times = torch.randn(n_points, 1)
    features = torch.rand(n_points, 2) * 2
    batch = torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts)
    )

    reference = FPSTokenizer(k_neighbors=8, **base).eval()

    def tokens(k_neighbors: int) -> Tensor:
        tokenizer = FPSTokenizer(k_neighbors=k_neighbors, **base).eval()
        tokenizer.load_state_dict(reference.state_dict())
        with torch.no_grad():
            return tokenizer(
                coords, features, batch, times, batch_size=len(counts)
            )[0]

    assert torch.allclose(tokens(10)[0], tokens(5)[0], atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
def test_low_precision_without_autocast(dtype: torch.dtype) -> None:
    """Test a module converted to low precision and run without autocast.

    This is what Lightning's `"16-true"` and `"bf16-true"` precision
    modes do. The Fourier position encoding evaluates its trigonometry
    in float32, so its output has to be cast back before reaching low-
    precision weights.
    """
    model = _model().to(dtype)
    batch = Batch.from_data_list(
        [
            Data(x=torch.randn(n, N_FEATURES).to(dtype))
            for n in (4, NUM_PATCHES * 3, 0)
        ]
    )
    output = model(batch)
    assert output.shape == (3, model.nb_outputs)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_coordinate_and_time_scaling() -> None:
    """Test that coordinates and time are rescaled and time is centred."""
    model = _model(xyz_scale=0.5, time_scale=30.0, center_time=True)
    x = torch.zeros(4, N_FEATURES)
    x[:, :3] = 2.0
    x[:, 3] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    coords, _, _, _, _ = model._prepare_inputs(Data(x=x))
    assert torch.allclose(coords[:, :3], torch.full((4, 3), 1.0))
    # Unit charge here, so centring subtracts the plain mean of 1.5 * 30.
    expected = (torch.tensor([0.0, 1.0, 2.0, 3.0]) - 1.5) * 30.0
    assert torch.allclose(coords[:, 3], expected)

    uncentred = _model(time_scale=30.0, center_time=False)
    coords, _, _, _, _ = uncentred._prepare_inputs(Data(x=x))
    assert torch.allclose(
        coords[:, 3], torch.tensor([0.0, 1.0, 2.0, 3.0]) * 30.0
    )


def test_invalid_configurations() -> None:
    """Test that misconfigurations fail fast with a clear message."""
    with pytest.raises(ValueError, match="divisible"):
        _model(token_dim=33, num_heads=2)
    with pytest.raises(ValueError, match="rotary embedding"):
        # head_dim = 4, below the minimum of 8 required by RoPE4D.
        Neptune(
            nb_inputs=N_FEATURES,
            token_dim=32,
            num_heads=8,
            **{
                k: v
                for k, v in TINY.items()
                if k not in ("token_dim", "num_heads")
            },
        )
    with pytest.raises(ValueError, match="exactly 3 columns"):
        _model(coordinate_columns=[0, 1])
    with pytest.raises(ValueError, match="charge_scaling"):
        _model(charge_scaling="sqrt")
    with pytest.raises(ValueError, match="out of range"):
        _model(time_column=99)


def test_model_config_round_trip(tmp_path: Any) -> None:
    """Test that `Neptune` survives a `ModelConfig` round trip."""
    path = os.path.join(str(tmp_path), "neptune.yml")
    model = _model(output_dim=5, pool_type="attention")
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
    backbone = Neptune(
        nb_inputs=data_representation.nb_outputs,
        coordinate_columns=[0, 1, 2],
        time_column=3,
        charge_column=None,
        xyz_scale=0.1,
        time_scale=10.5,
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
    nb_inputs = data_representation.nb_outputs
    batch = Batch.from_data_list(
        [Data(x=torch.randn(n, nb_inputs)) for n in (4, 20, 0)]
    )
    predictions = model(batch)
    assert len(predictions) == 1
    assert predictions[0].shape == (3, 1)


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
