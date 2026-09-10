"""Tests for DeepIce's query-tiled relative-attention path."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

from graphnet.models.gnn import DeepIce

N_FEATURES = 6

def _synth_batch(
    nev: int = 4,
    with_edges: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Batch:
    """Build a small batch of variable-length synthetic events."""
    g = torch.Generator().manual_seed(1)
    events = []
    for _ in range(nev):
        n = int(torch.randint(15, 50, (1,), generator=g))
        x = torch.randn(n, N_FEATURES, generator=g, dtype=dtype)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).to(dtype)
        data = Data(x=x)
        if with_edges:
            data.edge_index = knn_graph(x[:, :3].float(), k=8)
        data.n_pulses = torch.tensor([n])
        events.append(data)
    return Batch.from_data_list(events)

def _kwargs(include_dynedge: bool = False) -> dict:
    kw = dict(
        hidden_dim=128,
        seq_length=64,
        depth=2,
        head_size=32,
        depth_rel=3,
        n_rel=2,
        scaled_emb=True,
        include_dynedge=include_dynedge,
        n_features=N_FEATURES,
    )
    if include_dynedge:
        kw["dynedge_args"] = {
            "nb_inputs": N_FEATURES,
            "nb_neighbours": 8,
            "post_processing_layer_sizes": [64, 64],
            "activation_layer": "gelu",
            "add_norm_layer": True,
            "skip_readout": True,
        }
    return kw

@pytest.mark.parametrize("q_tile", [8, 64, 1000])
@pytest.mark.parametrize("include_dynedge", [False, True])
def test_tiled_bit_identical_to_dense(q_tile: int, include_dynedge: bool):
    """Same weights -> identical output, at any tile size."""
    torch.manual_seed(0)
    dense = DeepIce(**_kwargs(include_dynedge)).double().eval()
    tiled = (
        DeepIce(
            **_kwargs(include_dynedge),
            rel_attention="tiled",
            q_tile=q_tile,
        )
        .double()
        .eval()
    )
    # A stock (dense) checkpoint loads into a tiled model unchanged.
    tiled.load_state_dict(dense.state_dict())

    batch = _synth_batch(with_edges=include_dynedge)
    with torch.no_grad():
        out_dense = dense(batch)
        out_tiled = tiled(batch)
    assert (out_dense - out_tiled).abs().max().item() == 0.0

def test_tiled_gradients_match_dense():
    """Gradients agree, including the checkpointed per-tile recompute.

    Run in train mode (all dropouts / drop-paths are 0, so it is
    deterministic).
    """
    torch.manual_seed(0)
    dense = DeepIce(**_kwargs()).double().train()
    tiled = (
        DeepIce(**_kwargs(), rel_attention="tiled", q_tile=16)
        .double()
        .train()
    )
    tiled.load_state_dict(dense.state_dict())

    batch = _synth_batch()
    dense(batch).pow(2).sum().backward()
    tiled(batch).pow(2).sum().backward()

    dense_grads = dict(dense.named_parameters())
    max_err = 0.0
    for name, p in tiled.named_parameters():
        if p.grad is None:
            continue
        gd = dense_grads[name].grad
        assert gd is not None
        max_err = max(max_err, (p.grad - gd).abs().max().item())
    assert max_err < 1e-8, max_err

def test_tiled_trains_with_checkpoint():
    """Train mode: finite loss, gradients flow through checkpointed tiles."""
    torch.manual_seed(0)
    tiled = (
        DeepIce(**_kwargs(), rel_attention="tiled", q_tile=8)
        .double()
        .train()
    )
    out = tiled(_synth_batch())
    out.pow(2).sum().backward()
    gsum = sum(
        p.grad.abs().sum().item()
        for p in tiled.parameters()
        if p.grad is not None
    )
    assert torch.isfinite(out).all() and gsum > 0