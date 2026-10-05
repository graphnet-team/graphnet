"""Attention and transformer blocks used in graphnet models."""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import linear
from pytorch_lightning import LightningModule

from graphnet.models.components.embedding import apply_rope


class DropPath(LightningModule):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(
        self,
        drop_prob: float = 0.0,
    ):
        """Construct `DropPath`.

        Args:
            drop_prob: Probability of dropping a path during training.
                If 0.0, no paths are dropped. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(
        self,
        x: Tensor,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor; one drop decision per row of the batch dimension.
            doc_id: Optional `[N]` event index of each token of a packed
                `[1, N, D]` input; decisions are then drawn per event.
            num_docs: Number of events in `doc_id`.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        if doc_id is None:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0:
                random_tensor.div_(keep_prob)
            return x * random_tensor

        assert num_docs is not None, "`num_docs` is required with `doc_id`"
        event_mask = x.new_empty((num_docs, 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            event_mask = event_mask.div(keep_prob)
        token_mask = event_mask.index_select(0, doc_id).unsqueeze(0)
        return x * token_mask

    def extra_repr(self) -> str:
        """Return extra representation of the module."""
        return "p={}".format(self.drop_prob)


class Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x


class Attention_rel(LightningModule):
    """Attention mechanism with relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Attention_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            qk_scale: a scaling factor that multiplies the dot product of query
                and key vectors. Defaults to None. If None, computed as
                :math: `head_dim^(-1/2)`.
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
            attn_head_dim: the feature dimensionality of each attention head.
                Defaults to None. If None, computed as `dim // num_heads`.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or input_dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rel_pos_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        batch_size, event_length, _ = q.shape

        q = linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(batch_size, event_length, self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        k = linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(batch_size, k.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        v = linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(batch_size, v.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(batch_size, event_length, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_rel(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Block_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = Attention_rel(
            input_dim,
            num_heads,
            attn_drop=attn_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rel_pos_bias: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    kv,
                    kv,
                    rel_pos_bias=rel_pos_bias,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.gamma_1
                * self.drop_path(
                    self.attn(
                        xn,
                        kv,
                        kv,
                        rel_pos_bias=rel_pos_bias,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Block(LightningModule):
    """Transformer block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Construct 'Block'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the
                `MultiheadAttention` layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `MultiheadAttention`
                layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


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


class RotaryTransformerBlock(LightningModule):
    """Pre-norm transformer block with rotary attention.

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
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
    ):
        """Construct `RotaryTransformerBlock`.

        Args:
            d_model: Token dimension.
            nhead: Number of heads; `d_model` must be divisible by it.
            dim_feedforward: Inner dimension of the SwiGLU network.
            dropout: Dropout on the residual and feed-forward branches.
            layer_norm_eps: Epsilon of the RMSNorm layers.
            bias: Whether the attention projections carry a bias.
            ff_bias: Whether the feed-forward projections carry a bias.
            qk_norm: Whether to RMS-normalize queries and keys per head.
            drop_path_rate: Stochastic-depth rate of this block.
            layerscale_init: Initial value of the LayerScale parameters.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
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
        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(
        self,
        x: Tensor,
        rope_tables: Tuple[Tensor, Tensor],
        attn_mask: Optional[Tensor] = None,
        block_mask: Any = None,
        doc_id: Optional[Tensor] = None,
        num_docs: Optional[int] = None,
    ) -> Tensor:
        """Apply the block.

        Args:
            x: `[B, S, d_model]` token features.
            rope_tables: `(cos, sin)` tables from `RoPE4D`.
            attn_mask: Optional SDPA mask for the padded path.
            block_mask: `flex_attention` `BlockMask`; selects the packed path.
            doc_id: `[N]` event index per token on the packed path.
            num_docs: Number of events in `doc_id`.

        Returns:
            `[B, S, d_model]` updated token features.
        """
        batch_size, seq_length, _ = x.shape
        qkv = self.qkv_proj(self.norm1(x))
        qkv = qkv.reshape(batch_size, seq_length, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, S, head_dim]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, *rope_tables), apply_rope(k, *rope_tables)

        if block_mask is not None:
            attn = flex_attention(q, k, v, block_mask=block_mask)
        else:
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_length, -1)
        attn = self.dropout(self.gamma_1 * self.out_proj(attn))
        x = x + self.drop_path1(attn, doc_id, num_docs)
        ff = self.gamma_2 * self.ffn(self.norm2(x))
        return x + self.drop_path2(ff, doc_id, num_docs)


def _xavier_init(*linears: nn.Linear) -> None:
    """Apply Xavier-uniform weights and zero biases."""
    for layer in linears:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
