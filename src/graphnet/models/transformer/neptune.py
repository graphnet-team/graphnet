"""Neptune, a point transformer for neutrino events.

Paper: https://arxiv.org/abs/2510.01733. Reference implementation:
https://github.com/felixyu7/neptune.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.data import Data

from pytorch_lightning import LightningModule

from graphnet.models.components.embedding import FourierEncoder
from graphnet.models.components.neptune_layers import (
    AttentionPool,
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
)
from graphnet.models.components.tokenizers import FPSTokenizer
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import build_block_mask, pack, unpack

DEFAULT_ROPE_SCALES = [90.0, 90.0, 90.0, 1200.0]
# 90 down to 1.5 rad/unit over 20 bands, on each of x, y and z.
DEFAULT_POSITION_SCHEMA: Dict[int, Union[float, Tuple[float, float]]] = {
    c: (90.0, 60.0 ** (20 / 19)) for c in range(3)
}


class PointTransformerEncoder(LightningModule):
    """Rotary transformer over positioned tokens, pooled to one vector."""

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        pool_type: str = "mean",
        layerscale_init: float = 1e-5,
        attn_impl: str = "auto",
        position_encoding_schema: Optional[
            Dict[int, Union[float, Tuple[float, float]]]
        ] = None,
        position_encoding_bands: int = 20,
        rope_scales: Optional[List[float]] = None,
        rope_base: int = 60,
    ):
        """Construct `PointTransformerEncoder`.

        Args:
            d_model: Token dimension.
            depth: Number of encoder layers.
            num_heads: Number of attention heads.
            hidden_dim: Inner dimension of the SwiGLU feed-forward network.
            dropout: Dropout on the residual, feed-forward and position
                encoding paths.
            drop_path_rate: Stochastic-depth rate of the last layer, ramped
                linearly from zero.
            pool_type: `"mean"` (masked mean) or `"attention"` (learned
                query) pooling over tokens.
            layerscale_init: Initial value of the LayerScale parameters.
            attn_impl: `"padded"` runs SDPA on the padded `[B, S, D]` batch;
                `"packed"` runs block-diagonal `flex_attention` on the valid
                tokens only (padded on CPU under grad, where flex has no
                backward); `"auto"` packs on CUDA once the stack is compiled.
                All three give the same result.
            position_encoding_schema: `FourierEncoder` schema over the
                centroid columns `(x, y, z, t)`, as `{column: (highest
                frequency, span)}`. Time is encoded relatively by RoPE; add
                column 3 to also encode it absolutely.
            position_encoding_bands: Frequencies per encoded column.
            rope_scales: Highest rotary frequency per axis `(x, y, z, t)`.
            rope_base: Ratio of highest to lowest rotary frequency per axis.
        """
        super().__init__()
        if pool_type not in ("mean", "attention"):
            raise ValueError(f"Unknown pool_type '{pool_type}'")
        if attn_impl not in ("auto", "padded", "packed"):
            raise ValueError(f"Unknown attn_impl '{attn_impl}'")
        self.pool_type = pool_type
        self.attn_impl = attn_impl
        # Eager flex attention materializes the packed N x N, so "auto" packs
        # only once `Neptune.compile_layers` has run.
        self.packed_flex_ready = False

        def build_layer(rate: float) -> NeptuneTransformerEncoderLayer:
            return NeptuneTransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                drop_path_rate=rate,
                layerscale_init=layerscale_init,
                rope_scales=rope_scales or DEFAULT_ROPE_SCALES,
                rope_base=rope_base,
            )

        self.pos_encoder = FourierEncoder(
            position_encoding_schema or DEFAULT_POSITION_SCHEMA,
            seq_length=2 * position_encoding_bands,
            add_sequence_length=False,
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(self.pos_encoder.output_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.layers = NeptuneTransformerEncoder(
            build_layer, depth=depth, drop_path_rate=drop_path_rate
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        if pool_type == "attention":
            self.pool = AttentionPool(d_model)

    def _use_packed(self, src: Tensor, masks: Optional[Tensor]) -> bool:
        """Return whether the packed attention path applies."""
        if masks is None or self.attn_impl == "padded":
            return False
        if self.attn_impl == "packed":
            return src.is_cuda or not torch.is_grad_enabled()
        return src.is_cuda and self.packed_flex_ready

    def _encode(
        self, src: Tensor, centroids: Tensor, masks: Optional[Tensor]
    ) -> Tensor:
        """Run the encoder stack, packed or padded."""
        if self._use_packed(src, masks):
            assert masks is not None
            packed = pack(src, centroids, masks)
            if packed is not None:  # None: empty or densely filled batch
                tokens, cents, doc_id, pack_idx = packed
                out = self.layers(
                    tokens,
                    cents,
                    block_mask=build_block_mask(doc_id, tokens.shape[1]),
                    doc_id=doc_id,
                    num_docs=masks.shape[0],
                )
                return unpack(out, pack_idx, *masks.shape)
        padding = None if masks is None else ~masks
        return self.layers(src, centroids, src_key_padding_mask=padding)

    def forward(
        self,
        tokens: Tensor,
        centroids: Tensor,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode and pool a padded batch of tokens.

        Args:
            tokens: `[B, S, d_model]` token features.
            centroids: `[B, S, 4]` token positions `(x, y, z, t)`.
            masks: Optional bool `[B, S]`; True marks a valid token.

        Returns:
            `[B, d_model]` pooled event representation.
        """
        # Fourier phases in float32; low precision is too coarse for them.
        with torch.autocast(centroids.device.type, enabled=False):
            pos = self.pos_encoder(centroids.float())
        pos = self.pos_mlp(pos.to(self.pos_mlp[0].weight.dtype))
        if masks is not None:
            pos = pos * masks.unsqueeze(-1).to(pos.dtype)
        x = self.norm(self._encode(tokens + pos, centroids, masks))
        if self.pool_type == "attention":
            return self.pool(x, masks)
        if masks is None:
            return x.mean(dim=1)
        weights = masks.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class Neptune(GNN):
    """Neptune: a point transformer for neutrino events.

    Tokenizes each event by farthest point sampling (FPS) over its pulses,
    encodes the tokens with a rotary transformer and pools them to one
    vector. Use with `EdgelessGraph` and `NodesAsPulses`.

    Geometric defaults assume `IceCube86` standardization (one unit is
    0.5 km and 30 us); rescale `position_encoding_schema`, `rope_scales`
    and the tokenizer's `metric_time_scale` for other detectors. The charge
    column must hold physical charge, e.g. via
    `Detector(replace_with_identity=["charge"])`. For faster FPS,
    `pip install torch-fps`.
    """

    def __init__(
        self,
        input_feature_names: List[str],
        coordinate_columns: List[str],
        time_column: str,
        charge_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        num_patches: int = 128,
        d_model: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        output_dim: Optional[int] = None,
        pool_type: str = "mean",
        tokenizer_mlp_layers: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        layerscale_init: float = 1e-5,
        attn_impl: str = "auto",
        position_encoding_schema: Optional[
            Dict[int, Union[float, Tuple[float, float]]]
        ] = None,
        position_encoding_bands: int = 20,
        rope_scales: Optional[List[float]] = None,
        rope_base: int = 60,
        compile_encoder: bool = False,
    ):
        """Construct `Neptune`.

        Args:
            input_feature_names: Names of the columns of `data.x`.
            coordinate_columns: Names of the `(x, y, z)` columns.
            time_column: Name of the time column.
            charge_column: Name of the charge column. None for detectors
                without charge (e.g. Prometheus): every pulse then has unit
                charge.
            feature_columns: Further per-pulse features for the tokenizer.
                Defaults to every column not named above.
            num_patches: Maximum number of tokens per event.
            d_model: Token dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads; `d_model / num_heads` must
                be even and at least 8.
            hidden_dim: Inner dimension of the SwiGLU feed-forward network.
            dropout: Dropout used throughout the model.
            drop_path_rate: Stochastic-depth rate of the last layer.
            output_dim: Readout width (`nb_outputs`); defaults to `d_model`.
            pool_type: See `PointTransformerEncoder`.
            tokenizer_mlp_layers: Hidden sizes of the tokenizer's per-pulse
                MLP; defaults to `[256, 512, 768]`.
            tokenizer_kwargs: Further `FPSTokenizer` arguments, e.g.
                `assign_mode`, `lloyd_iters` or `metric_time_scale`.
            layerscale_init: Initial value of the LayerScale parameters.
            attn_impl: See `PointTransformerEncoder`.
            position_encoding_schema: See `PointTransformerEncoder`.
            position_encoding_bands: See `PointTransformerEncoder`.
            rope_scales: See `PointTransformerEncoder`.
            rope_base: See `PointTransformerEncoder`.
            compile_encoder: Call `compile_layers` at construction.
        """
        if len(coordinate_columns) != 3:
            raise ValueError(
                f"coordinate_columns must name 3 columns, got "
                f"{coordinate_columns}"
            )
        named = [*coordinate_columns, time_column]
        if charge_column is not None:
            named.append(charge_column)
        if feature_columns is None:
            feature_columns = [
                name for name in input_feature_names if name not in named
            ]
        for name in named + feature_columns:
            if name not in input_feature_names:
                raise ValueError(
                    f"'{name}' is not in input_feature_names "
                    f"{input_feature_names}"
                )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})"
            )
        head_dim = d_model // num_heads
        if head_dim % 2 != 0 or head_dim < 8:
            raise ValueError(
                f"d_model / num_heads = {head_dim}, but the 4D rotary "
                "embedding needs an even head dimension of at least 8"
            )
        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        reserved = set(tokenizer_kwargs) & {
            "feature_dim",
            "max_tokens",
            "token_dim",
            "mlp_layers",
            "dropout",
            "charge_col",
        }
        if reserved:
            raise ValueError(
                f"tokenizer_kwargs must not set {sorted(reserved)}; Neptune "
                "sets them"
            )
        super().__init__(len(input_feature_names), output_dim or d_model)

        index = input_feature_names.index
        self._coordinate_index = [index(name) for name in coordinate_columns]
        self._time_index = [index(time_column)]
        self._charge_index = (
            None if charge_column is None else index(charge_column)
        )
        self._feature_index = [index(name) for name in feature_columns]

        self.tokenizer = FPSTokenizer(
            feature_dim=1 + len(feature_columns),
            max_tokens=num_patches,
            token_dim=d_model,
            mlp_layers=tokenizer_mlp_layers,
            dropout=dropout,
            **tokenizer_kwargs,
        )
        self.encoder = PointTransformerEncoder(
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            pool_type=pool_type,
            layerscale_init=layerscale_init,
            attn_impl=attn_impl,
            position_encoding_schema=position_encoding_schema,
            position_encoding_bands=position_encoding_bands,
            rope_scales=rope_scales,
            rope_base=rope_base,
        )
        # +2 inputs: log token count and log total charge, which pooling
        # normalizes away.
        self.head = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.nb_outputs),
        )
        if compile_encoder:
            self.compile_layers()

    def compile_layers(
        self, mode: str = "default", dynamic: bool = True
    ) -> None:
        """Compile the transformer stack in place with `torch.compile`.

        Worth 2-4x on GPU, and enables the `"auto"` packed attention path.
        State-dict keys are unchanged.

        Args:
            mode: `torch.compile` mode.
            dynamic: Compile with dynamic shapes.
        """
        self.encoder.layers.compile(mode=mode, dynamic=dynamic)
        self.encoder.packed_flex_ready = True

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass.

        Args:
            data: Batch of events.

        Returns:
            `[num_graphs, nb_outputs]` event representations.
        """
        x, batch, batch_size = data.x, data.batch, data.num_graphs
        if self._charge_index is None:
            charge = x.new_ones(x.shape[0])
        else:
            charge = x[:, self._charge_index].clamp(min=0)
        features = torch.cat(
            [torch.log1p(charge).unsqueeze(-1), x[:, self._feature_index]],
            dim=1,
        )
        tokens, centroids, masks = self.tokenizer(
            x[:, self._coordinate_index],
            features,
            batch,
            x[:, self._time_index],
            batch_size=batch_size,
        )
        pooled = self.encoder(tokens, centroids, masks)

        total_charge = torch.zeros(batch_size, device=x.device)
        total_charge = total_charge.index_add_(0, batch, charge.float())
        extras = torch.stack(
            [torch.log1p(masks.sum(dim=1).float()), torch.log1p(total_charge)],
            dim=-1,
        )
        return self.head(torch.cat([pooled, extras.to(pooled.dtype)], dim=-1))
