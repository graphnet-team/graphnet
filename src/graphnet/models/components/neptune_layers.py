"""Layer classes for the Neptune point-transformer backbone."""

from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.attention.flex_attention import flex_attention

from pytorch_lightning import LightningModule

from graphnet.models.components.attention_blocks import DropPath


class SwiGLU(LightningModule):
    """Feed-forward block with a SwiGLU gate and fused input projection."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Construct `SwiGLU`.

        Args:
            dim: Input and output dimension.
            hidden_dim: Inner dimension of the gated projection.
            dropout: Dropout applied to the block output.
            bias: Whether the projections carry a bias.
        """
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        _xavier_init(self.w13, self.w2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward transform."""
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.dropout(self.w2(torch.nn.functional.silu(a) * b))


class RoPE4D(LightningModule):
    """Axis-aligned rotary position embedding over `(x, y, z, t)`.

    Each axis gets at least one rotation plane, the rest are dealt out
    round-robin; pair `j` rotates dimensions `(j, dim / 2 + j)`. See
    https://arxiv.org/abs/2504.06308.
    """

    freqs: Tensor
    coord_select: Tensor

    def __init__(
        self,
        dim: int,
        scales: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        base: int = 10000,
    ):
        """Construct `RoPE4D`.

        Args:
            dim: Head dimension; even and at least 8.
            scales: Highest frequency per axis, in radians per input unit.
            base: Frequencies on each axis span `scale / base` to `scale`.
        """
        super().__init__()
        if dim % 2 != 0 or dim < 8:
            raise ValueError(f"RoPE4D needs an even dim >= 8 (got {dim})")
        if len(scales) != 4:
            raise ValueError(f"RoPE4D expects 4 scales (got {len(scales)})")
        allocation = [1, 1, 1, 1]
        for i in range(dim // 2 - 4):
            allocation[i % 4] += 1

        freqs = []
        for n_planes, scale in zip(allocation, scales):
            exponents = torch.arange(n_planes, dtype=torch.float32)
            exponents = exponents / max(n_planes - 1, 1)
            freqs.append((1.0 / base) * (base**exponents) * scale)
        self.register_buffer("freqs", torch.cat(freqs))
        coord_select: List[int] = []
        for axis, n_planes in enumerate(allocation):
            coord_select += [axis] * n_planes
        self.register_buffer("coord_select", torch.tensor(coord_select))

    def compute_tables(
        self, coords: Tensor, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return `(cos, sin)` tables `[B, 1, S, dim / 2]` for `coords`.

        Args:
            coords: `[B, S, 4]` coordinates.
            dtype: Dtype to cast the tables to; the trig runs in float32.
        """
        coords = coords.float().index_select(-1, self.coord_select)
        angles = (coords * self.freqs).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        tables: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Rotate `[B, H, S, dim]` queries or keys by their coordinates.

        Args:
            x: `[B, H, S, dim]` queries or keys.
            coords: `[B, S, 4]` coordinates.
            tables: Optional precomputed output of :meth:`compute_tables`.
        """
        cos, sin = tables or self.compute_tables(coords, dtype=x.dtype)
        x1, x2 = x.to(torch.promote_types(x.dtype, cos.dtype)).chunk(2, -1)
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.to(x.dtype)


class AttentionPool(LightningModule):
    """Pool a sequence to one vector, attending from a learned query."""

    def __init__(self, dim: int):
        """Construct `AttentionPool`.

        Args:
            dim: Feature dimension of the sequence and of the output.
        """
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * (dim**-0.5))
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pool `[B, S, D]` to `[B, D]`; fully masked rows give zero.

        Args:
            x: `[B, S, D]` sequence.
            mask: Optional bool `[B, S]`; True marks a valid element.
        """
        q = self.q.expand(x.shape[0], -1, -1)
        k, v = self.kv(x).chunk(2, dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) * (x.shape[-1] ** -0.5)
        if mask is not None:
            # A finite fill keeps fully masked rows NaN-free; zeroed below.
            attn = attn.masked_fill(
                ~mask.unsqueeze(1), torch.finfo(attn.dtype).min
            )
        out = self.proj(torch.bmm(attn.softmax(dim=-1), v).squeeze(1))
        if mask is not None:
            out = out * mask.any(dim=1, keepdim=True).to(out.dtype)
        return out


class NeptuneTransformerEncoderLayer(LightningModule):
    """Pre-norm encoder layer with 4D rotary attention.

    Uses RMSNorm, a fused QKV projection, per-head query/key norm, LayerScale
    and a SwiGLU feed-forward. There is no attention-matrix dropout, which
    `flex_attention` cannot express, so the packed and padded paths match.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        ff_bias: bool = False,
        qk_norm: bool = True,
        rope_scales: Sequence[float] = (90.0, 90.0, 90.0, 1200.0),
        rope_base: int = 60,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
    ):
        """Construct `NeptuneTransformerEncoderLayer`.

        Args:
            d_model: Token dimension.
            nhead: Number of heads; `d_model / nhead` must be even and >= 8.
            dim_feedforward: Inner dimension of the SwiGLU network.
            dropout: Dropout on the residual and feed-forward branches.
            layer_norm_eps: Epsilon of the RMSNorm layers.
            bias: Whether the attention projections carry a bias.
            ff_bias: Whether the feed-forward projections carry a bias.
            qk_norm: Whether to RMS-normalize queries and keys per head.
            rope_scales: See :class:`RoPE4D`.
            rope_base: See :class:`RoPE4D`.
            drop_path_rate: Stochastic-depth rate of this layer.
            layerscale_init: Initial value of the LayerScale parameters.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        _xavier_init(self.qkv_proj, self.out_proj)
        self.norm1 = nn.RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.RMSNorm(d_model, eps=layer_norm_eps)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=layer_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=layer_norm_eps)
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout, bias=ff_bias)
        self.rope = RoPE4D(self.head_dim, scales=rope_scales, base=rope_base)
        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    @staticmethod
    def prepare_attention_mask(
        key_padding_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Turn a `[B, S]` padding mask (True = pad) into an SDPA mask.

        Fully padded rows attend everywhere to stay NaN-free; pooling
        zeroes them afterwards.
        """
        if key_padding_mask is None:
            return None
        allow = ~key_padding_mask.bool()
        allow = allow | ~allow.any(dim=-1, keepdim=True)
        return allow[:, None, None, :]

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        rope_tables: Optional[Tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Apply one encoder layer.

        Args:
            src: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token coordinates for the rotary embedding.
            src_key_padding_mask: Bool `[B, S]`, True marks padding; used
                only when neither `attn_mask` nor `block_mask` is given.
            rope_tables: Optional shared `(cos, sin)` tables.
            attn_mask: Optional prebuilt SDPA mask.
            block_mask: `flex_attention` `BlockMask`; selects the packed path.
            doc_id: `[N]` event index per token on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` updated token features.
        """
        batch_size, seq_length, _ = src.shape
        qkv = self.qkv_proj(self.norm1(src))
        qkv = qkv.reshape(batch_size, seq_length, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, S, head_dim]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q, centroids, tables=rope_tables)
        k = self.rope(k, centroids, tables=rope_tables)

        if block_mask is not None:
            attn = flex_attention(q, k, v, block_mask=block_mask)
        else:
            if attn_mask is None:
                attn_mask = self.prepare_attention_mask(src_key_padding_mask)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_length, -1)
        attn = self.dropout(self.gamma_1 * self.out_proj(attn))
        x = src + self.drop_path1(attn, doc_id, num_docs)
        ff = self.gamma_2 * self.ffn(self.norm2(x))
        return x + self.drop_path2(ff, doc_id, num_docs)


class NeptuneTransformerEncoder(LightningModule):
    """Stack of :class:`NeptuneTransformerEncoderLayer`.

    The rotary tables and attention mask are built once and shared by
    all layers.
    """

    def __init__(
        self,
        layer_factory: Callable[[float], NeptuneTransformerEncoderLayer],
        depth: int,
        drop_path_rate: float = 0.0,
    ):
        """Construct `NeptuneTransformerEncoder`.

        Args:
            layer_factory: Maps a stochastic-depth rate to a new layer.
            depth: Number of layers.
            drop_path_rate: Stochastic-depth rate of the last layer, ramped
                linearly from zero.
        """
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        rates = [drop_path_rate]
        if depth > 1:
            rates = [drop_path_rate * i / (depth - 1) for i in range(depth)]
        self.layers = nn.ModuleList([layer_factory(r) for r in rates])

    def forward(
        self,
        src: Tensor,
        centroids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Run the stack; arguments as in the layer's `forward`."""
        first = cast(NeptuneTransformerEncoderLayer, self.layers[0])
        rope_tables = first.rope.compute_tables(centroids, dtype=src.dtype)
        attn_mask = None
        if block_mask is None:
            attn_mask = first.prepare_attention_mask(src_key_padding_mask)
        for layer in self.layers:
            src = layer(
                src,
                centroids,
                rope_tables=rope_tables,
                attn_mask=attn_mask,
                block_mask=block_mask,
                doc_id=doc_id,
                num_docs=num_docs,
            )
        return src


def _xavier_init(*linears: nn.Linear) -> None:
    """Apply Xavier-uniform weights and zero biases."""
    for linear in linears:
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
