"""Unit tests for the sinusoidal input embeddings."""

import pytest
import torch

from graphnet.models.components.embedding import (
    FourierEncoder,
    FourierEncoderEPJC,
    SinusoidalPosEmb,
)

BATCH, LENGTH, N_FEATURES, SEQ = 3, 7, 6, 8


def _inputs() -> tuple:
    torch.manual_seed(0)
    x = torch.randn(BATCH, LENGTH, N_FEATURES)
    seq_length = torch.tensor([LENGTH, LENGTH - 2, LENGTH - 5])
    return x, seq_length


def test_per_column_matches_strided_block() -> None:
    """One embedding per column equals `FourierEncoderEPJC`'s xyz block.

    The encoding is elementwise, so embedding a `[B, K, 3]` slice and
    flattening lays the channels out exactly as concatenating three
    single-column embeddings does. This is what lets `FourierEncoder` drop
    the positional slicing without changing the representation.
    """
    x, _ = _inputs()
    sin = SinusoidalPosEmb(dim=SEQ)

    block = sin(4096 * x[:, :, :3]).flatten(-2)
    per_column = torch.cat([sin(4096 * x[:, :, c]) for c in range(3)], -1)

    assert torch.equal(block, per_column)


def test_schema_reproduces_epjc_embedding() -> None:
    """A schema of the EPJC columns gives its pre-projection embedding."""
    x, seq_length = _inputs()
    encoder = FourierEncoder(
        schema={0: 4096.0, 1: 4096.0, 2: 4096.0, 4: 1024.0, 3: 4096.0},
        seq_length=SEQ,
    )
    epjc = FourierEncoderEPJC(seq_length=SEQ, n_features=5)

    length = torch.log10(seq_length.to(dtype=x.dtype))
    expected = torch.cat(
        [
            epjc.sin_emb(4096 * x[:, :, :3]).flatten(-2),
            epjc.sin_emb(1024 * x[:, :, 4]),
            epjc.sin_emb(4096 * x[:, :, 3]),
            epjc.sin_emb2(length).unsqueeze(1).expand(-1, LENGTH, -1),
        ],
        -1,
    )

    output = encoder(x, seq_length)
    assert torch.equal(output, expected)
    assert output.shape[-1] == encoder.output_dim


def test_multiplier_is_applied() -> None:
    """The schema's multiplier reaches the sinusoidal ladder."""
    x, _ = _inputs()

    coarse = FourierEncoder(
        schema={3: 1.0}, seq_length=SEQ, add_sequence_length=False
    )(x)
    fine = FourierEncoder(
        schema={3: 4096.0}, seq_length=SEQ, add_sequence_length=False
    )(x)

    assert coarse.shape[-1] == SEQ
    assert not torch.allclose(coarse, fine)


def test_column_order_follows_schema() -> None:
    """Output blocks are ordered by the schema, not by column index."""
    x, _ = _inputs()

    forward = FourierEncoder(
        schema={0: 1.0, 1: 1.0}, seq_length=SEQ, add_sequence_length=False
    )(x)
    reversed_ = FourierEncoder(
        schema={1: 1.0, 0: 1.0}, seq_length=SEQ, add_sequence_length=False
    )(x)

    assert torch.equal(forward[..., :SEQ], reversed_[..., SEQ:])


def test_empty_schema_rejected() -> None:
    """An encoder that embeds nothing is a configuration error."""
    with pytest.raises(ValueError):
        FourierEncoder(schema={}, seq_length=SEQ)


def test_sequence_length_required_when_embedded() -> None:
    """`seq_length` is mandatory when its embedding is requested."""
    x, _ = _inputs()
    encoder = FourierEncoder(schema={0: 1.0}, seq_length=SEQ)

    with pytest.raises(ValueError):
        encoder(x)
