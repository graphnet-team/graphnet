"""Implementation of the Neptune point-transformer architecture.

Neptune -- "a N Efficient Point Transformer for Ultrarelativistic
Neutrino Events" -- treats an event as an irregular 4D point cloud and
reconstructs it with a rotary-attention transformer over sampled tokens.

- Paper: https://arxiv.org/abs/2510.01733
- Reference implementation: https://github.com/felixyu7/neptune
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch_geometric.data import Data

from pytorch_lightning import LightningModule

from graphnet.models.components.embedding import FourierPositionEncoder
from graphnet.models.components.layers import (
    AttentionPool,
    NeptuneTransformerEncoder,
    NeptuneTransformerEncoderLayer,
    RMSNorm,
)
from graphnet.models.components.tokenizers import FPSTokenizer
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import (
    FLEX_AVAILABLE,
    build_block_mask,
    pack,
    unpack,
)

DEFAULT_FOURIER_AXIS_SCALES = [1.0, 1.0, 1.0]
DEFAULT_ROPE_SCALES = [180.0, 180.0, 180.0, 40.0]


class PointTransformerEncoder(LightningModule):
    """Rotary transformer encoder over tokens with known positions.

    Adds a Fourier absolute position encoding of the token centroids to the
    token features, runs the encoder stack, and pools the result to one
    vector per event. Attention runs either over the padded `[B, S, D]`
    batch or, on CUDA, over a packed `[1, N, D]` sequence with a
    block-diagonal `flex_attention` mask -- mathematically identical, but
    without pushing padded slots through every projection.
    """

    def __init__(
        self,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        pool_type: str = "mean",
        layerscale_init: float = 1e-5,
        attn_impl: str = "auto",
        fourier_num_bands: int = 20,
        fourier_freq_min: float = 3.0,
        fourier_freq_max: float = 180.0,
        fourier_axis_scales: Optional[List[float]] = None,
        rope_scales: Optional[List[float]] = None,
        rope_base: int = 60,
    ):
        """Construct `PointTransformerEncoder`.

        Args:
            token_dim: Token dimension.
            num_layers: Depth of the encoder stack.
            num_heads: Number of attention heads.
            hidden_dim: Inner dimension of the SwiGLU feed-forward network.
            dropout: Dropout on the residual, feed-forward, and position
                encoding paths.
            drop_path_rate: Stochastic-depth rate of the last layer, ramped
                linearly from zero over the stack.
            pool_type: `"mean"` for masked mean pooling, or `"attention"`
                for cross-attention pooling with a learned query.
            layerscale_init: Initial value of the LayerScale parameters.
            attn_impl: `"auto"`, `"padded"`, or `"packed"`. Purely a
                throughput choice; all three are mathematically equivalent.
            fourier_num_bands: Frequency bands of the absolute position
                encoding.
            fourier_freq_min: Lowest position-encoding frequency, in radians
                per coordinate unit.
            fourier_freq_max: Highest position-encoding frequency, in
                radians per coordinate unit.
            fourier_axis_scales: Per-axis multipliers for the position
                encoding. Defaults to `[1.0, 1.0, 1.0]`.
            rope_scales: Per-axis rotary frequency scales for
                `(x, y, z, t)`. Defaults to `[180.0, 180.0, 180.0, 40.0]`.
            rope_base: Rotary frequency span; see
                :class:`~graphnet.models.components.layers.RoPE4D`.
        """
        super().__init__()
        if pool_type not in ("mean", "attention"):
            raise ValueError(
                f"pool_type must be 'mean' or 'attention', got '{pool_type}'"
            )
        if attn_impl not in ("auto", "padded", "packed"):
            raise ValueError(
                "attn_impl must be 'auto', 'padded' or 'packed', got "
                f"'{attn_impl}'"
            )
        if fourier_axis_scales is None:
            fourier_axis_scales = list(DEFAULT_FOURIER_AXIS_SCALES)
        if rope_scales is None:
            rope_scales = list(DEFAULT_ROPE_SCALES)

        self.token_dim = token_dim
        self.pool_type = pool_type
        # "auto": packed flex on CUDA, padded SDPA on CPU (where the packed
        # batch is a single sequence anyway). "padded": always padded.
        # "packed": force packed where supported. Dense batches and overflow
        # fall back to padded inside `pack`.
        self.attn_impl = attn_impl
        # The "auto" packed path is only safe once a compiled flex path is
        # active: eager `flex_attention` materializes the full packed N x N
        # and can run out of memory. Set True by `compile_layers` on success,
        # cleared on failure or fallback.
        self.packed_flex_ready = False

        def build_layer(rate: float) -> NeptuneTransformerEncoderLayer:
            return NeptuneTransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                drop_path_rate=rate,
                layerscale_init=layerscale_init,
                rope_scales=rope_scales,
                rope_base=rope_base,
            )

        # Absolute position encoding: log-spaced Fourier features on
        # (x, y, z). Time is dropped -- events are time-centred, so absolute
        # time is close to meaningless and the rotary embedding handles
        # relative time. The encoder slices time off the 4D centroids
        # internally.
        self.abs_pos_encoder = FourierPositionEncoder(
            token_dim,
            in_dim=3,
            num_bands=fourier_num_bands,
            freq_min=fourier_freq_min,
            freq_max=fourier_freq_max,
            axis_scales=fourier_axis_scales,
            dropout=dropout,
        )
        self.layers = NeptuneTransformerEncoder(
            build_layer,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
        )
        self.norm = RMSNorm(token_dim)

        if pool_type == "attention":
            self.pool = AttentionPool(token_dim)

    def _use_packed(self, src: Tensor, masks: Optional[Tensor]) -> bool:
        """Return whether the packed attention path should be used."""
        if masks is None or not FLEX_AVAILABLE or self.attn_impl == "padded":
            return False
        if self.attn_impl == "packed":
            # `flex_attention` has no CPU backward; under grad on CPU it
            # raises at the forward call. Fall back to padded there even when
            # the packed path is forced.
            return src.is_cuda or not torch.is_grad_enabled()
        # "auto": GPU packs only when the compiled flex path is active. CPU
        # and uncompiled runs stay on padded SDPA.
        return src.is_cuda and self.packed_flex_ready

    def _encode(
        self, src: Tensor, centroids: Tensor, masks: Optional[Tensor]
    ) -> Tensor:
        """Run the encoder stack, packed or padded.

        Packing uses data-dependent shapes and stays eager; only
        `self.layers` is ever compiled.
        """
        if self._use_packed(src, masks):
            assert masks is not None
            packed = pack(src, centroids, masks)
            if packed is not None:  # None: empty or dense -> padded below
                tokens, cents, doc_id, pack_idx = packed
                block_mask = build_block_mask(doc_id, tokens.shape[1])
                batch_size, seq_length = masks.shape
                # doc_id (= pack_idx // S) indexes original events, so
                # num_docs=B lets DropPath sample one decision per event.
                out = self.layers(
                    tokens,
                    cents,
                    block_mask=block_mask,
                    doc_id=doc_id,
                    num_docs=batch_size,
                )
                return unpack(out, pack_idx, batch_size, seq_length)
        padding = (~masks) if masks is not None else None
        return self.layers(src, centroids, src_key_padding_mask=padding)

    def forward(
        self,
        tokens: Tensor,
        centroids: Tensor,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode and pool a padded batch of tokens.

        Args:
            tokens: `[B, S, token_dim]` token features.
            centroids: `[B, S, 4]` token positions `(x, y, z, t)`.
            masks: Optional bool `[B, S]`; True marks a valid token.

        Returns:
            `[B, token_dim]` pooled event representation.
        """
        # No cast to `tokens.dtype`: the position encoder computes in float32
        # anyway, and a bfloat16 round-trip of km-scale coordinates costs up
        # to ~0.7 rad of phase at the default `fourier_freq_max`.
        centroid_emb = self.abs_pos_encoder(centroids)
        if masks is not None:
            centroid_emb = centroid_emb * masks.to(
                dtype=centroid_emb.dtype
            ).unsqueeze(-1)
        x = self._encode(tokens + centroid_emb, centroids, masks)
        x = self.norm(x)

        if self.pool_type == "attention":
            return self.pool(x, masks)

        if masks is None:
            return x.mean(dim=1)
        weights = masks.to(dtype=x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom.squeeze(1)


class Neptune(GNN):
    """Neptune: an efficient point transformer for neutrino events.

    Tokenizes an event by farthest point sampling over the 4D
    `(x, y, z, t)` point cloud, encodes the tokens with a rotary-attention
    transformer, pools to a single event vector, and applies an MLP readout.
    Unlike sequence-based transformers in GraphNeT it needs neither edges nor
    a cap on the number of pulses: use it with
    :class:`~graphnet.models.data_representation.graphs.graphs.EdgelessGraph`
    and
    :class:`~graphnet.models.data_representation.graphs.nodes.nodes.NodesAsPulses`.

    **Input contract.** Neptune's geometric priors are expressed in physical
    units: `fourier_freq_min` / `fourier_freq_max` and the first three
    `rope_scales` are radians per kilometre, the fourth is radians per
    microsecond, and `tokenizer_kwargs["metric_time_scale"]` is km/us. The
    tokenizer additionally treats feature column 0 as `log1p` of the physical
    charge. A GraphNeT `Detector` standardizes columns however its experiment
    chose, so this class converts:

    - the coordinate columns are multiplied by `xyz_scale` to reach km,
    - the time column is multiplied by `time_scale` to reach microseconds and
      then centred per event, and
    - the charge column is converted from `charge_scaling` to `log1p` and
      placed first in the feature vector.

    Ready-made settings for detectors shipped with GraphNeT:

    - `IceCube86`: `xyz_scale=0.5`, `time_scale=30.0`,
      `charge_scaling="log10"`.
    - `IceCubeKaggle`: as above plus `charge_scale=3.0`, since its charge
      column holds `log10(charge) / 3` rather than `log10(charge)`.
    - `Prometheus` / `ORCA150SuperDense`: `xyz_scale=0.1`,
      `time_scale=10.5`, `charge_column=None` (no charge is recorded).

    The defaults for `fourier_freq_*`, `rope_scales`, and
    `metric_time_scale` are tuned to IceCube's roughly 1 km by 10 us scale.
    For a substantially smaller detector, rescale them accordingly.

    Note that `self.training` changes the *tokenization*, not only dropout:
    farthest point sampling starts from a random point during training as an
    augmentation, and from a deterministic one at evaluation.
    """

    def __init__(
        self,
        nb_inputs: int,
        coordinate_columns: Optional[List[int]] = None,
        time_column: int = 3,
        charge_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        xyz_scale: float = 1.0,
        time_scale: float = 1.0,
        center_time: bool = True,
        charge_scaling: str = "log1p",
        charge_scale: float = 1.0,
        num_patches: int = 128,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        output_dim: Optional[int] = None,
        pool_type: str = "mean",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        layerscale_init: float = 1e-5,
        attn_impl: str = "auto",
        fourier_num_bands: int = 20,
        fourier_freq_min: float = 3.0,
        fourier_freq_max: float = 180.0,
        fourier_axis_scales: Optional[List[float]] = None,
        rope_scales: Optional[List[float]] = None,
        rope_base: int = 60,
        compile_encoder: bool = False,
    ):
        """Construct `Neptune`.

        Args:
            nb_inputs: Number of columns in `data.x`, i.e.
                `data_representation.nb_outputs`.
            coordinate_columns: Columns of `data.x` holding `(x, y, z)`.
                Defaults to `[0, 1, 2]`.
            time_column: Column of `data.x` holding the pulse time.
            charge_column: Column of `data.x` holding the charge, in the
                scaling given by `charge_scaling`. None if the detector
                records no charge, in which case every pulse is given unit
                charge and the charge-weighted statistics reduce to
                unweighted ones.
            feature_columns: Columns of `data.x` passed to the tokenizer in
                addition to charge. Defaults to every column not used as a
                coordinate, time, or charge.
            xyz_scale: Multiplier converting the standardized coordinates to
                kilometres.
            time_scale: Multiplier converting the standardized time to
                microseconds.
            center_time: Whether to subtract a per-event charge-weighted mean
                time. Strongly recommended: the rotary time frequencies are
                relative, and the absolute position encoding drops time on
                the assumption that events are centred. This also absorbs any
                constant offset in the detector's time standardization, which
                is why a single `time_scale` multiplier suffices. Note that
                the reference implementation centres on the charge-weighted
                *median*; the mean is used here because it needs no sort.
            charge_scaling: Scaling the detector applied to the charge
                column, one of `"log1p"`, `"log10"`, or `"linear"`.
            charge_scale: Multiplier applied to the charge column before the
                `charge_scaling` is undone, for detectors that rescale the
                logarithm. `IceCubeKaggle`, for instance, stores
                `log10(charge) / 3`, which needs `charge_scale=3.0`.
            num_patches: Maximum number of tokens per event.
            token_dim: Transformer hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads. `token_dim / num_heads`
                must be even and at least 8.
            hidden_dim: Inner dimension of the SwiGLU feed-forward network.
            dropout: Dropout used throughout the model.
            drop_path_rate: Stochastic-depth rate of the last transformer
                layer, ramped linearly from zero over the stack.
            output_dim: Width of the readout, i.e. `nb_outputs`. Defaults to
                `token_dim`. A GraphNeT `Task` adds its own linear layer on
                top; to reproduce the reference model exactly, set this to
                the task's output width and construct the `Task` with
                `disable_affine=True`.
            pool_type: `"mean"` or `"attention"` pooling over tokens.
            tokenizer_kwargs: Extra keyword arguments forwarded to
                :class:`~graphnet.models.components.tokenizers.FPSTokenizer`,
                for example `assign_mode`, `lloyd_iters`, `knn_pool`,
                `k_neighbors`, `metric_time_scale`, or `mlp_layers`.
            layerscale_init: Initial value of the LayerScale parameters.
            attn_impl: `"auto"`, `"padded"`, or `"packed"` attention path.
            fourier_num_bands: Frequency bands of the absolute position
                encoding.
            fourier_freq_min: Lowest position-encoding frequency, rad/km.
            fourier_freq_max: Highest position-encoding frequency, rad/km.
            fourier_axis_scales: Per-axis multipliers for the position
                encoding. Defaults to `[1.0, 1.0, 1.0]`.
            rope_scales: Per-axis rotary frequency scales for
                `(x, y, z, t)`, in rad/km and rad/us. Defaults to
                `[180.0, 180.0, 180.0, 40.0]`.
            rope_base: Rotary frequency span; frequencies on each axis run
                from `scale / rope_base` to `scale`.
            compile_encoder: Whether to `torch.compile` the transformer
                stack. Worth 2-4x throughput on GPU, and also what enables
                the packed attention path. Off by default so that importing
                and training the model has no compilation side effect; a
                compilation failure at runtime falls back to the uncompiled
                encoder rather than crashing.
        """
        if coordinate_columns is None:
            coordinate_columns = [0, 1, 2]
        if len(coordinate_columns) != 3:
            raise ValueError(
                "coordinate_columns must name exactly 3 columns, got "
                f"{coordinate_columns}"
            )
        if charge_scaling not in ("log1p", "log10", "linear"):
            raise ValueError(
                "charge_scaling must be 'log1p', 'log10' or 'linear', got "
                f"'{charge_scaling}'"
            )
        claimed = set(coordinate_columns) | {time_column}
        if charge_column is not None:
            claimed.add(charge_column)
        if feature_columns is None:
            feature_columns = [i for i in range(nb_inputs) if i not in claimed]
        for column in [*coordinate_columns, time_column, *feature_columns]:
            if not 0 <= column < nb_inputs:
                raise ValueError(
                    f"column {column} is out of range for nb_inputs="
                    f"{nb_inputs}"
                )
        if charge_column is not None and not 0 <= charge_column < nb_inputs:
            raise ValueError(
                f"charge_column {charge_column} is out of range for "
                f"nb_inputs={nb_inputs}"
            )
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        head_dim = token_dim // num_heads
        if head_dim % 2 != 0 or head_dim < 8:
            raise ValueError(
                f"token_dim / num_heads = {head_dim}, but the 4D rotary "
                "embedding requires an even head dimension of at least 8"
            )

        if output_dim is None:
            output_dim = token_dim
        super().__init__(nb_inputs, output_dim)

        self._coordinate_columns = coordinate_columns
        self._time_column = time_column
        self._charge_column = charge_column
        self._feature_columns = feature_columns
        self._xyz_scale = xyz_scale
        self._time_scale = time_scale
        self._center_time = center_time
        self._charge_scaling = charge_scaling
        self._charge_scale = charge_scale

        tokenizer_cfg: Dict[str, Any] = dict(tokenizer_kwargs or {})
        mlp_layers = tokenizer_cfg.pop("mlp_layers", [256, 512, 768])
        tokenizer_dropout = tokenizer_cfg.pop("dropout", dropout)
        # Charge is always placed first by `forward`, so the tokenizer's own
        # default of column 0 is correct and must not be overridden.
        tokenizer_cfg.pop("charge_col", None)

        self.tokenizer = FPSTokenizer(
            feature_dim=1 + len(feature_columns),
            max_tokens=num_patches,
            token_dim=token_dim,
            mlp_layers=mlp_layers,
            dropout=tokenizer_dropout,
            **tokenizer_cfg,
        )

        self.encoder = PointTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            pool_type=pool_type,
            layerscale_init=layerscale_init,
            attn_impl=attn_impl,
            fourier_num_bands=fourier_num_bands,
            fourier_freq_min=fourier_freq_min,
            fourier_freq_max=fourier_freq_max,
            fourier_axis_scales=fourier_axis_scales,
            rope_scales=rope_scales,
            rope_base=rope_base,
        )

        # The two extra inputs are event-level scalars: log token
        # multiplicity and log total charge. Pooling normalizes event size
        # and total light yield away, yet both carry direct signal -- energy
        # above all.
        self.head = nn.Sequential(
            nn.Linear(token_dim + 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, output_dim),
        )

        self.encoder_compiled = False
        if compile_encoder:
            self.encoder_compiled = self.compile_layers()

    def compile_layers(
        self, mode: str = "default", dynamic: bool = True
    ) -> bool:
        """Compile the transformer stack for faster training and inference.

        Only `self.encoder.layers` is compiled: packing uses data-dependent
        shapes and the tokenizer has data-dependent control flow, both of
        which would force graph breaks. `dynamic=True` lets one graph serve
        both the varying packed length `[1, N, D]` and the padded batch
        `[B, S, D]`. Compilation is applied in place via `nn.Module.compile`,
        so `state_dict` keys are unchanged and existing checkpoints stay
        loadable, and it is lazy, so it targets the model's final device.

        Args:
            mode: `torch.compile` mode.
            dynamic: Whether to compile with dynamic shapes.

        Returns:
            True if compilation was set up, False if it is unavailable, in
            which case the model runs uncompiled.
        """
        if not hasattr(torch, "compile") or not hasattr(
            self.encoder.layers, "compile"
        ):
            warnings.warn(
                "torch.compile is unavailable in this PyTorch build; "
                "running the Neptune encoder uncompiled."
            )
            self.encoder_compiled = False
            return False
        try:
            self.encoder.layers.compile(mode=mode, dynamic=dynamic)
        except Exception as exc:  # setup failure; tracing is lazy, so rare
            warnings.warn(
                "Neptune encoder torch.compile setup failed "
                f"({type(exc).__name__}: {exc}); falling back to the "
                "uncompiled encoder."
            )
            self._revert_compilation()
            return False
        self.encoder_compiled = True
        self.encoder.packed_flex_ready = True
        return True

    def _revert_compilation(self) -> None:
        """Drop back to the eager encoder and the padded attention path."""
        self.encoder_compiled = False
        # Route "auto" back to padded SDPA: eager flex would materialize the
        # whole packed N x N, so the retry must not stay on the packed path.
        self.encoder.packed_flex_ready = False
        try:
            self.encoder.layers._compiled_call_impl = None
        except Exception:
            pass

    def _run_encoder(
        self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]
    ) -> Tensor:
        """Run the encoder, falling back to eager on a compile failure.

        The guard stays armed beyond the first forward because lazy
        dynamic compilation traces each distinct path on first use; the
        packed path, for instance, only compiles when a sparse batch
        first reaches it.
        """
        if not self.encoder_compiled:
            return self.encoder(tokens, centroids, masks)
        try:
            return self.encoder(tokens, centroids, masks)
        except Exception as exc:  # compile or backend failure at runtime
            warnings.warn(
                "Neptune encoder torch.compile failed at runtime "
                f"({type(exc).__name__}: {exc}); falling back to the "
                "uncompiled encoder and padded attention for the rest of "
                "this process. `Neptune.encoder_compiled` records the "
                "downgrade."
            )
            self._revert_compilation()
            return self.encoder(tokens, centroids, masks)

    def _physical_charge(self, column: Tensor) -> Tensor:
        """Undo the detector's charge scaling to recover physical charge."""
        scaled = column * self._charge_scale
        if self._charge_scaling == "log1p":
            charge = torch.expm1(scaled)
        elif self._charge_scaling == "log10":
            charge = torch.pow(
                torch.tensor(10.0, dtype=scaled.dtype, device=scaled.device),
                scaled,
            )
        else:
            charge = scaled
        return charge.clamp(min=0)

    @staticmethod
    def _event_mean(
        values: Tensor, weights: Tensor, batch: Tensor, batch_size: int
    ) -> Tensor:
        """Return the per-event weighted mean of `values`.

        Events whose weights sum to zero -- every pulse carrying no
        charge -- fall back to an unweighted mean rather than collapsing
        to zero.
        """
        zeros = torch.zeros(
            batch_size, dtype=values.dtype, device=values.device
        )
        weight_sum = zeros.clone().index_add_(0, batch, weights)
        count = zeros.clone().index_add_(0, batch, torch.ones_like(weights))
        degenerate = weight_sum <= 0
        weights = torch.where(
            degenerate.index_select(0, batch),
            torch.ones_like(weights),
            weights,
        )
        weight_sum = torch.where(degenerate, count, weight_sum)
        numerator = zeros.clone().index_add_(0, batch, values * weights)
        return numerator / weight_sum.clamp(min=1e-12)

    def _prepare_inputs(
        self, data: Data
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Map a GraphNeT `Data` object onto Neptune's input contract.

        Args:
            data: Batch of events, with `data.x` standardized by a
                `Detector`.

        Returns:
            `(coords [N, 4] in km and centred microseconds, features [N, F]
            with `log1p` charge first, batch [N], physical charge [N],
            batch_size)`.
        """
        x = data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            batch_size = 1
        else:
            batch = batch.long()
            # `num_graphs` is authoritative: inferring the batch size from
            # `batch.max()` would silently drop trailing events with no
            # pulses and misalign predictions against labels.
            batch_size = int(getattr(data, "num_graphs", 0)) or (
                int(batch.max()) + 1 if batch.numel() else 0
            )

        xyz = x[:, self._coordinate_columns] * self._xyz_scale
        time = x[:, self._time_column] * self._time_scale

        if self._charge_column is None:
            q_phys = torch.ones_like(time)
        else:
            q_phys = self._physical_charge(x[:, self._charge_column])

        if self._center_time and time.numel():
            # Accumulate in float32: under a low-precision "-true" run this
            # reduction spans every pulse of an event, and half-precision
            # sums lose meaningful accuracy there. A no-op in float32.
            offset = self._event_mean(
                time.float(), q_phys.float(), batch, batch_size
            ).index_select(0, batch)
            time = time - offset.to(time.dtype)

        coords = torch.cat([xyz, time.unsqueeze(-1)], dim=1)
        features = torch.cat(
            [torch.log1p(q_phys).unsqueeze(-1), x[:, self._feature_columns]],
            dim=1,
        )
        return coords, features, batch, q_phys, batch_size

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass.

        Args:
            data: Batch of events.

        Returns:
            `[batch_size, nb_outputs]` event representations.
        """
        coords, features, batch, q_phys, batch_size = self._prepare_inputs(
            data
        )

        tokens, centroids, masks = self.tokenizer(
            coords[:, :3],
            features,
            batch,
            coords[:, 3:],
            batch_size=batch_size,
        )
        global_feat = self._run_encoder(tokens, centroids, masks)

        # Event-level scalars that pooling normalizes away: the number of
        # valid tokens and the total physical charge of the event.
        n_tokens = masks.sum(dim=1).float()
        total_q = torch.zeros(batch_size, device=tokens.device).index_add_(
            0, batch, q_phys.float()
        )
        extras = torch.stack(
            [torch.log1p(n_tokens), torch.log1p(total_q)], dim=-1
        ).to(global_feat.dtype)

        return self.head(torch.cat([global_feat, extras], dim=-1))
