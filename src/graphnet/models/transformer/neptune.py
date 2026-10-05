"""Neptune, a point transformer for neutrino events.

Paper: https://arxiv.org/abs/2510.01733. Reference implementation:
https://github.com/felixyu7/neptune.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.data import Data

from graphnet.models.components.attention_blocks import (
    AttentionPool,
    RotaryTransformerBlock,
)
from graphnet.models.components.embedding import FourierEncoder, RoPE4D
from graphnet.models.components.tokenizers import FPSTokenizer
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import build_block_mask, pack, unpack

DEFAULT_ROPE_SCALES = [90.0, 90.0, 90.0, 1200.0]
# 90 down to 1.5 rad/unit over 20 bands, on each of x, y and z.
DEFAULT_POSITION_SCHEMA: Dict[int, Union[float, Tuple[float, float]]] = {
    c: (90.0, 60.0 ** (20 / 19)) for c in range(3)
}


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
    `pip install "torch-fps==0.6.*"`.
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
        position_encoding_schema: Optional[
            Dict[int, Union[float, Tuple[float, float]]]
        ] = None,
        position_encoding_bands: int = 20,
        rope_scales: Optional[List[float]] = None,
        rope_base: int = 60,
        compile_encoder: bool = True,
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
                be even and at least 8 for the 4D rotary embedding.
            hidden_dim: Inner dimension of the SwiGLU feed-forward network.
            dropout: Dropout used throughout the model.
            drop_path_rate: Stochastic-depth rate of the last layer, ramped
                linearly from zero.
            output_dim: Readout width (`nb_outputs`); defaults to `d_model`.
            pool_type: `"mean"` (masked mean) or `"attention"` (learned
                query) pooling over tokens.
            tokenizer_mlp_layers: Hidden sizes of the tokenizer's per-pulse
                MLP; defaults to `[256, 512, 768]`.
            tokenizer_kwargs: Further `FPSTokenizer` arguments, e.g.
                `assign_mode`, `lloyd_iters` or `metric_time_scale`.
            layerscale_init: Initial value of the LayerScale parameters.
            position_encoding_schema: `FourierEncoder` schema over the
                centroid columns `(x, y, z, t)`, as `{column: (highest
                frequency, span)}`. Time is encoded relatively by RoPE; add
                column 3 to also encode it absolutely.
            position_encoding_bands: Frequencies per encoded column.
            rope_scales: Highest rotary frequency per axis `(x, y, z, t)`.
            rope_base: Ratio of highest to lowest rotary frequency per axis.
            compile_encoder: `torch.compile` the transformer layers, 2-4x
                faster on GPU, where it also enables block-diagonal packed
                attention over only the valid tokens. Turn off for CPU-only
                or debugging runs.
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
        if pool_type not in ("mean", "attention"):
            raise ValueError(f"Unknown pool_type '{pool_type}'")
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
        self.rope = RoPE4D(
            d_model // num_heads,
            scales=rope_scales or DEFAULT_ROPE_SCALES,
            base=rope_base,
        )
        self.layers = nn.ModuleList(
            RotaryTransformerBlock(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                drop_path_rate=rate,
                layerscale_init=layerscale_init,
            )
            for rate in torch.linspace(0, drop_path_rate, depth).tolist()
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.pool = (
            AttentionPool(d_model) if pool_type == "attention" else None
        )
        # +2 inputs: log token count and log total charge, which pooling
        # normalizes away.
        self.head = nn.Sequential(
            nn.Linear(d_model + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.nb_outputs),
        )
        # Eager flex attention materializes the packed N x N, so packing
        # needs the compiled layers.
        self.compile_encoder = compile_encoder
        if compile_encoder:
            for layer in self.layers:
                layer.compile(dynamic=True)

    def _encode(self, x: Tensor, centroids: Tensor, masks: Tensor) -> Tensor:
        """Run the transformer layers, packed when compiled on GPU."""
        packed = None
        if self.compile_encoder and x.is_cuda:
            packed = pack(x, centroids, masks)  # None if packing won't pay
        kwargs: Dict[str, Any]
        if packed is None:
            # Fully padded rows attend everywhere to stay NaN-free; pooling
            # zeroes them afterwards.
            allow = masks | ~masks.any(dim=-1, keepdim=True)
            kwargs = {"attn_mask": allow[:, None, None, :]}
        else:
            x, centroids, doc_id, pack_idx = packed
            kwargs = {
                "block_mask": build_block_mask(doc_id, x.shape[1]),
                "doc_id": doc_id,
                "num_docs": masks.shape[0],
            }
        rope_tables = self.rope(centroids, x.dtype)
        for layer in self.layers:
            x = layer(x, rope_tables, **kwargs)
        return x if packed is None else unpack(x, pack_idx, *masks.shape)

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

        # Fourier phases in float32; low precision is too coarse for them.
        with torch.autocast(centroids.device.type, enabled=False):
            pos = self.pos_encoder(centroids.float())
        pos = self.pos_mlp(pos.to(self.pos_mlp[0].weight.dtype))
        tokens = tokens + pos * masks.unsqueeze(-1).to(pos.dtype)
        tokens = self.norm(self._encode(tokens, centroids, masks))
        if self.pool is not None:
            pooled = self.pool(tokens, masks)
        else:
            weights = masks.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * weights).sum(dim=1)
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)

        total_charge = torch.zeros(batch_size, device=x.device)
        total_charge = total_charge.index_add_(0, batch, charge.float())
        extras = torch.stack(
            [torch.log1p(masks.sum(dim=1).float()), torch.log1p(total_charge)],
            dim=-1,
        )
        return self.head(torch.cat([pooled, extras.to(pooled.dtype)], dim=-1))
