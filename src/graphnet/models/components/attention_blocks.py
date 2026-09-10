"""Attention and transformer blocks used in graphnet models."""

from typing import Optional

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.functional import linear
from pytorch_lightning import LightningModule


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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

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
