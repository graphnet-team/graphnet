"""Learnable tokenizers turning point clouds into transformer tokens."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.functional import Tensor

from pytorch_lightning import LightningModule

from graphnet.models.components.fps import (
    farthest_point_sampling_with_assign,
    farthest_point_sampling_with_knn,
    nearest_assign,
)


class FPSTokenizer(LightningModule):
    """Tokenize a batch of point clouds by farthest point sampling.

    Ported from the Neptune reference implementation
    (https://github.com/felixyu7/neptune, MIT licence), described in
    https://arxiv.org/abs/2510.01733.

    The pipeline is:

    1. A per-point MLP lifts the input features.
    2. Events with at most `max_tokens` points are passed through
       one-token-per-point: no sampling, no pooling.
    3. Larger events are reduced to `max_tokens` centroids by farthest point
       sampling in a 4D `(x, y, z, t)` metric, then pooled per token either
       by nearest-centroid (Voronoi) assignment -- every point contributes to
       exactly one token, so nothing is discarded -- or, with
       `assign_mode="knn"`, by a k-nearest-neighbour gather around each
       centroid.
    4. Four per-token summary scalars (multiplicity, total charge, time
       spread, spatial RMS radius) are appended to the pooled features.
    5. A second MLP refines the tokens, applied in both branches so the depth
       is consistent.

    The forward pass performs exactly one host synchronization (a device-to-
    host copy of the batch indices) and uses no data-dependent-shape
    operations such as `nonzero` or boolean indexing: small and large events
    are routed with precomputed index tensors, and only the large subset is
    ever padded, and only to its own maximum length. One very bright event
    therefore does not inflate the memory of the whole batch.
    """

    N_EXTRA = 4  # multiplicity, total charge, time spread, RMS radius

    metric_scale: Tensor

    def __init__(
        self,
        feature_dim: int,
        max_tokens: int = 128,
        token_dim: int = 768,
        mlp_layers: Optional[List[int]] = None,
        k_neighbors: int = 8,
        dropout: float = 0.0,
        knn_pool: str = "max",
        rel_pos_hidden: int = 64,
        charge_weighted_mean: bool = True,
        charge_col: int = 0,
        assign_mode: str = "voronoi",
        metric_time_scale: float = 0.3,
        lloyd_iters: int = 0,
    ):
        """Construct `FPSTokenizer`.

        Args:
            feature_dim: Number of input features per point.
            max_tokens: Maximum number of tokens produced per event.
            token_dim: Output token dimension.
            mlp_layers: Hidden sizes of the per-point MLP. Defaults to
                `[256, 512, 768]`.
            k_neighbors: Neighbours gathered per centroid when
                `assign_mode="knn"`.
            dropout: Dropout inside both MLPs.
            knn_pool: `"max"` for masked max pooling, or `"max_mean"` to
                concatenate a charge-weighted mean, doubling the pooled
                width.
            rel_pos_hidden: Hidden size of the relative-geometry encoder.
            charge_weighted_mean: Whether the mean pooling and the summary
                scalars are weighted by charge. The weight is the `log1p`
                charge as stored in `charge_col`, not the physical charge:
                a deliberate compression that stops one very bright sensor
                from dominating a token. Only the total-charge summary
                scalar and the Lloyd refinement use physical charge.
            charge_col: Column of the feature tensor holding `log1p` of the
                point's total charge. The tokenizer applies `expm1` to it to
                recover the physical charge, so any other scaling breaks the
                charge-weighted statistics.
            assign_mode: `"voronoi"` or `"knn"`, see the class docstring.
            metric_time_scale: Weight of the time axis in the sampling and
                assignment metric only, in coordinate units per time unit.
                With positions in km and time in microseconds, the default of
                0.3 km/us is roughly the speed of light and balances space
                against time; 0.22 is the photon group velocity in ice, and
                1.0 leaves the raw units. At raw units the time axis carries
                almost all of the 4D metric variance on real events, so
                selection would cluster nearly purely by arrival time. The
                returned centroids, relative offsets, and summary scalars
                always stay in raw units -- the scale changes token
                membership, never the representation.
            lloyd_iters: Number of charge-weighted Lloyd (k-means)
                refinement steps applied to the sampled centroids. Ignored
                in `"knn"` mode. Farthest point sampling solves a
                k-center-style objective (equal cell *radius*), which
                over-compresses dense bright cores and spends tokens on stray
                peripheral points; Lloyd refinement moves centroids towards
                the charge-weighted vector-quantization optimum. With
                `lloyd_iters > 0` the centroids become virtual points
                (charge-weighted cell means in raw units over the final
                membership) rather than points of the cloud, so relative
                offsets are zero-mean per cell. Cells may then be empty,
                giving an all-zero token whose multiplicity scalar is 0 while
                its mask entry stays True.
        """
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 512, 768]
        if assign_mode not in ("voronoi", "knn"):
            raise ValueError(
                f"assign_mode must be 'voronoi' or 'knn', got '{assign_mode}'"
            )
        if lloyd_iters < 0:
            raise ValueError(f"lloyd_iters must be >= 0, got {lloyd_iters}")
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.knn_pool = knn_pool
        self.charge_weighted_mean = charge_weighted_mean
        self.charge_col = charge_col
        self.assign_mode = assign_mode
        self.metric_time_scale = metric_time_scale
        self.lloyd_iters = lloyd_iters

        # MLP 1: per-point feature extraction.
        mlp1: List[nn.Module] = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp1 += [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        mlp1 += [nn.Linear(in_dim, token_dim)]
        self.mlp1 = nn.Sequential(*mlp1)

        # MLP 2: token refinement, always applied. The input width doubles
        # with max+mean pooling; the summary scalars are appended either way.
        pooled_dim = 2 * token_dim if knn_pool == "max_mean" else token_dim
        self.mlp2 = nn.Sequential(
            nn.Linear(pooled_dim + self.N_EXTRA, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

        # Relative-geometry encoder: embeds each point's (dx, dy, dz, dt)
        # offset from its token centroid and adds it to the point feature
        # before pooling, so a token encodes local cluster shape (otherwise
        # lost in a pure feature max/mean). Runs only on the large-event path.
        self.rel_encoder = nn.Sequential(
            nn.Linear(4, rel_pos_hidden),
            nn.GELU(),
            nn.Linear(rel_pos_hidden, token_dim),
        )

        # A buffer rather than a per-forward host tensor, which would cost a
        # pageable host-to-device copy plus a stream sync on every call.
        self.register_buffer(
            "metric_scale",
            torch.tensor([1.0, 1.0, 1.0, float(metric_time_scale)]),
            persistent=False,
        )

    @staticmethod
    def _route_subset(
        rows_dev: Tensor,
        csub: Tensor,
        starts: Tensor,
        n_sub: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build index tensors routing a subset out of batch-sorted arrays.

        The inputs are device slices of the single packed routing transfer,
        so there is no per-call host-to-device sync and no `nonzero`.

        Args:
            rows_dev: `[B_sub]` original event index of each subset member.
            csub: `[B_sub]` point count of each subset member.
            starts: `[B_sub]` offset of each subset member in the sorted
                arrays.
            n_sub: Total number of points in the subset.
            device: Device to build the indices on.

        Returns:
            `(seg, local, src)`: the subset-event id of each point, its
            position within that event, and its row in the batch-sorted flat
            arrays.
        """
        seg = torch.repeat_interleave(
            torch.arange(rows_dev.numel(), device=device),
            csub,
            output_size=n_sub,
        )
        local = (
            torch.arange(n_sub, device=device)
            - (torch.cumsum(csub, 0) - csub)[seg]
        )
        src = starts[seg] + local
        return seg, local, src

    def forward(
        self,
        coords: Tensor,
        features: Tensor,
        batch_ids: Tensor,
        times: Tensor,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Tokenize a ragged batch of point clouds.

        Args:
            coords: `[N, 3]` spatial coordinates `(x, y, z)`.
            features: `[N, F]` per-point features.
            batch_ids: `[N]` event index of each point.
            times: `[N, 1]` time coordinate of each point.
            batch_size: True number of events. Strongly recommended: without
                it the batch size is inferred as `max(batch_ids) + 1`, which
                silently drops trailing events with no points and misaligns
                the output with the labels.

        Returns:
            `(tokens [B, max_tokens, token_dim], centroids [B, max_tokens, 4]
            in `(x, y, z, t)`, masks [B, max_tokens] bool)`.
        """
        device = coords.device
        dtype_p = coords.dtype
        dtype_f = features.dtype
        n_tokens = self.max_tokens
        token_dim = self.token_dim

        if coords.numel() == 0:
            n_events = batch_size or 0
            return (
                torch.zeros(
                    (n_events, n_tokens, token_dim),
                    device=device,
                    dtype=dtype_f,
                ),
                torch.zeros(
                    (n_events, n_tokens, 4), device=device, dtype=dtype_p
                ),
                torch.zeros(
                    (n_events, n_tokens), device=device, dtype=torch.bool
                ),
            )

        batch_idx = batch_ids.long()
        points4 = torch.cat([coords[:, :3], times], dim=-1)  # raw units
        point_feats = self.mlp1(features)
        charge = features[:, self.charge_col]  # log1p charge

        # The single host sync: one device-to-host copy of the indices.
        # `bincount` on a CUDA tensor would itself sync, since it reads
        # max(batch_idx) on the host to size its output, so counting happens
        # on the CPU copy instead.
        counts_list = torch.bincount(
            batch_idx.cpu(), minlength=(batch_size or 0)
        ).tolist()
        n_events = len(counts_list)

        # Sort points by event so each event is a contiguous run.
        sort_idx = torch.argsort(batch_idx, stable=True)
        p_sorted = points4.index_select(0, sort_idx)
        f_sorted = point_feats.index_select(0, sort_idx)
        q_sorted = charge.index_select(0, sort_idx)

        # Host-side split into small (at most `max_tokens` points: every
        # point is a token) and large (sampling plus pooling) events. All
        # routing below derives from these lists -- no boolean indexing, no
        # `nonzero`, no further syncs.
        offsets: List[int] = [0] * n_events
        acc = 0
        for i, count in enumerate(counts_list):
            offsets[i] = acc
            acc += count
        small_rows = [i for i, c in enumerate(counts_list) if c <= n_tokens]
        large_rows = [i for i, c in enumerate(counts_list) if c > n_tokens]
        counts_s = [counts_list[i] for i in small_rows]
        counts_l = [counts_list[i] for i in large_rows]
        n_small, n_large = len(small_rows), len(large_rows)
        pts_small, pts_large = sum(counts_s), sum(counts_l)

        # All routing metadata goes up in one pinned, non-blocking transfer;
        # a per-list `torch.tensor(..., device=...)` would each be a pageable
        # copy plus a full stream sync.
        routing_host = torch.tensor(
            small_rows
            + counts_s
            + [offsets[r] for r in small_rows]
            + large_rows
            + counts_l
            + [offsets[r] for r in large_rows],
            dtype=torch.long,
        )
        if device.type == "cuda":
            routing_host = routing_host.pin_memory()
        routing = routing_host.to(device, non_blocking=True)
        rows_s_dev = routing[:n_small]
        csub_s = routing[n_small : 2 * n_small]
        starts_s = routing[2 * n_small : 3 * n_small]
        off = 3 * n_small
        rows_l_dev = routing[off : off + n_large]
        csub_l = routing[off + n_large : off + 2 * n_large]
        starts_l_abs = routing[off + 2 * n_large : off + 3 * n_large]

        pooled_dim = (
            2 * token_dim if self.knn_pool == "max_mean" else token_dim
        )
        feat_dtype = point_feats.dtype  # autocast-aware compute dtype

        pre = torch.zeros(
            n_events,
            n_tokens,
            pooled_dim + self.N_EXTRA,
            device=device,
            dtype=feat_dtype,
        )
        cents_out = torch.zeros(
            n_events, n_tokens, 4, device=device, dtype=dtype_p
        )
        masks = torch.zeros(
            n_events, n_tokens, device=device, dtype=torch.bool
        )

        if n_small > 0:
            self._tokenize_small(
                pre,
                cents_out,
                masks,
                rows_s_dev,
                csub_s,
                starts_s,
                pts_small,
                f_sorted,
                p_sorted,
                q_sorted,
                pooled_dim,
                feat_dtype,
                dtype_p,
            )

        if n_large > 0:
            self._tokenize_large(
                pre,
                cents_out,
                masks,
                rows_l_dev,
                csub_l,
                starts_l_abs,
                pts_large,
                max(counts_l),
                f_sorted,
                p_sorted,
                q_sorted,
                feat_dtype,
                dtype_p,
            )

        # MLP 2: token refinement, applied to both branches.
        tokens = self.mlp2(pre)

        # Zero out padded positions.
        tokens = tokens * masks.unsqueeze(-1)
        cents_out = cents_out * masks.unsqueeze(-1)

        return tokens.to(dtype_f), cents_out, masks

    def _tokenize_small(
        self,
        pre: Tensor,
        cents_out: Tensor,
        masks: Tensor,
        rows_dev: Tensor,
        csub: Tensor,
        starts: Tensor,
        n_pts: int,
        f_sorted: Tensor,
        p_sorted: Tensor,
        q_sorted: Tensor,
        pooled_dim: int,
        feat_dtype: torch.dtype,
        dtype_p: torch.dtype,
    ) -> None:
        """Fill `pre`/`cents_out`/`masks` for events with few points.

        Every point becomes its own token: no sampling, no padding, and no
        pooling.
        """
        device = pre.device
        n_tokens = self.max_tokens
        n_sub = rows_dev.numel()
        seg, local, src = self._route_subset(
            rows_dev, csub, starts, n_pts, device
        )
        feats = f_sorted.index_select(0, src)
        points = p_sorted.index_select(0, src)
        charge = q_sorted.index_select(0, src)
        dest = seg * n_tokens + local  # unique: local < counts <= n_tokens

        pooled = (
            torch.cat([feats, feats], dim=-1)
            if self.knn_pool == "max_mean"
            else feats
        )
        # A single point: multiplicity 1 (log1p(1) = log 2), its own charge,
        # and zero spreads.
        zeros = torch.zeros_like(charge)
        extras = torch.stack(
            [
                torch.full_like(charge, 0.6931471805599453),
                charge,
                zeros,
                zeros,
            ],
            dim=-1,
        )
        pre_sub = torch.zeros(
            n_sub * n_tokens,
            pooled_dim + self.N_EXTRA,
            device=device,
            dtype=feat_dtype,
        )
        pre_sub.index_copy_(
            0, dest, torch.cat([pooled, extras.to(feat_dtype)], dim=-1)
        )
        cents_sub = torch.zeros(
            n_sub * n_tokens, 4, device=device, dtype=dtype_p
        )
        cents_sub.index_copy_(0, dest, points)

        pre.index_copy_(0, rows_dev, pre_sub.view(n_sub, n_tokens, -1))
        cents_out.index_copy_(0, rows_dev, cents_sub.view(n_sub, n_tokens, 4))
        masks.index_copy_(
            0,
            rows_dev,
            torch.arange(n_tokens, device=device)[None, :] < csub[:, None],
        )

    def _tokenize_large(
        self,
        pre: Tensor,
        cents_out: Tensor,
        masks: Tensor,
        rows_dev: Tensor,
        csub: Tensor,
        starts_abs: Tensor,
        n_pts: int,
        n_max: int,
        f_sorted: Tensor,
        p_sorted: Tensor,
        q_sorted: Tensor,
        feat_dtype: torch.dtype,
        dtype_p: torch.dtype,
    ) -> None:
        """Fill `pre`/`cents_out`/`masks` for events with many points.

        Centroids come from farthest point sampling; features are pooled
        per token by Voronoi assignment or a kNN gather.
        """
        device = pre.device
        n_tokens = self.max_tokens
        n_sub = rows_dev.numel()
        seg, local, src = self._route_subset(
            rows_dev, csub, starts_abs, n_pts, device
        )
        feats = f_sorted.index_select(0, src)
        points = p_sorted.index_select(0, src)  # raw units
        charge = q_sorted.index_select(0, src)

        # Metric copy for sampling and assignment: float32, time axis
        # scaled. Padding stays zero (finite) -- the backend masks it but
        # must not see NaNs.
        points_m = points.float() * self.metric_scale
        padded_m = torch.zeros(n_sub * n_max, 4, device=device)
        dest_pad = seg * n_max + local
        padded_m.index_copy_(0, dest_pad, points_m)
        padded_m = padded_m.view(n_sub, n_max, 4)
        valid = torch.arange(n_max, device=device)[None, :] < csub[:, None]

        # Offsets of each subset event within the flat subset arrays.
        starts_sub = torch.cumsum(csub, 0) - csub

        if self.assign_mode == "voronoi":
            pre_sub, cents_sub = self._pool_voronoi(
                padded_m,
                valid,
                points,
                points_m,
                feats,
                charge,
                seg,
                dest_pad,
                starts_sub,
                csub,
                n_sub,
                n_max,
                feat_dtype,
            )
        else:
            pre_sub, cents_sub = self._pool_knn(
                padded_m,
                valid,
                points,
                feats,
                charge,
                dest_pad,
                starts_sub,
                n_sub,
                n_max,
                feat_dtype,
            )

        pre.index_copy_(0, rows_dev, pre_sub.view(n_sub, n_tokens, -1))
        cents_out.index_copy_(
            0, rows_dev, cents_sub.view(n_sub, n_tokens, 4).to(dtype_p)
        )
        masks.index_copy_(
            0,
            rows_dev,
            torch.ones(n_sub, n_tokens, device=device, dtype=torch.bool),
        )

    def _pool_voronoi(
        self,
        padded_m: Tensor,
        valid: Tensor,
        points: Tensor,
        points_m: Tensor,
        feats: Tensor,
        charge: Tensor,
        seg: Tensor,
        dest_pad: Tensor,
        starts_sub: Tensor,
        csub: Tensor,
        n_sub: int,
        n_max: int,
        feat_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Pool by nearest-centroid assignment, optionally Lloyd-refined."""
        device = points.device
        n_tokens = self.max_tokens
        token_dim = self.token_dim
        n_cells = n_sub * n_tokens

        # Fused sampling and assignment: the sampling loop already computes
        # every point-to-centroid distance, so the initial Voronoi
        # assignment is free. Runs in float32 regardless of autocast.
        # `random_start` gates the augmentation on training mode, so
        # evaluation always tokenizes an event identically. `validate=False`
        # is safe here: counts > max_tokens holds by construction of the
        # subset.
        fps_idx, assign_pad = farthest_point_sampling_with_assign(
            padded_m,
            valid,
            n_tokens,
            random_start=self.training,
            validate=False,
            assume_finite=True,
        )
        cent_flat = (starts_sub[:, None] + fps_idx).reshape(-1)
        cents_raw = points.index_select(0, cent_flat)
        cents_m = points_m.index_select(0, cent_flat).view(n_sub, n_tokens, 4)

        # Every point joins exactly one token (coverage is 1 by
        # construction); assignment happens in the scaled metric.
        assign = assign_pad.reshape(-1).index_select(0, dest_pad)
        cell = seg * n_tokens + assign

        q_phys = torch.expm1(charge.float().clamp(min=0))

        if self.lloyd_iters > 0:
            # Charge-weighted Lloyd refinement towards the
            # vector-quantization optimum. Empty cells keep their previous
            # centroid; assignment uses standard alternating updates in the
            # scaled metric.
            cm_flat = cents_m.reshape(n_cells, 4)
            cell4 = cell.unsqueeze(-1).expand(-1, 4)
            for _ in range(self.lloyd_iters):
                wsum = torch.zeros(n_cells, device=device).scatter_add_(
                    0, cell, q_phys
                )
                csum = torch.zeros(n_cells, 4, device=device).scatter_add_(
                    0, cell4, points_m * q_phys[:, None]
                )
                cm_flat = torch.where(
                    (wsum > 0)[:, None],
                    csum / wsum.clamp(min=1e-9)[:, None],
                    cm_flat,
                )
                assign = nearest_assign(
                    points_m, cm_flat, starts_sub, csub, n_max, n_tokens
                )
                cell = seg * n_tokens + assign
                cell4 = cell.unsqueeze(-1).expand(-1, 4)
            # Raw-unit centroids: charge-weighted cell means over the FINAL
            # membership, giving zero-mean relative offsets per cell. Empty
            # cells fall back to their sampled point position.
            wsum = torch.zeros(n_cells, device=device).scatter_add_(
                0, cell, q_phys
            )
            craw = torch.zeros(n_cells, 4, device=device).scatter_add_(
                0, cell4, points * q_phys[:, None]
            )
            cents_raw = torch.where(
                (wsum > 0)[:, None],
                (craw / wsum.clamp(min=1e-9)[:, None]).to(cents_raw.dtype),
                cents_raw,
            )

        rel = points - cents_raw.index_select(0, cell)  # raw units
        hidden = feats + self.rel_encoder(rel).to(feat_dtype)

        cell_t = cell.unsqueeze(-1).expand(-1, token_dim)
        pooled = torch.full(
            (n_cells, token_dim),
            float("-inf"),
            device=device,
            dtype=hidden.dtype,
        )
        pooled = pooled.scatter_reduce(
            0, cell_t, hidden, reduce="amax", include_self=True
        )
        pooled = torch.where(
            torch.isfinite(pooled), pooled, torch.zeros_like(pooled)
        )

        # float32 accumulators for the weighted mean and summary scalars.
        weight = (
            charge.float()
            if self.charge_weighted_mean
            else torch.ones_like(charge, dtype=torch.float32)
        )
        wsum = torch.zeros(n_cells, device=device).scatter_add_(
            0, cell, weight
        )
        wsafe = wsum.clamp(min=1e-6)

        if self.knn_pool == "max_mean":
            weighted = hidden.float() * weight[:, None]
            hsum = torch.zeros(n_cells, token_dim, device=device).scatter_add_(
                0, cell_t, weighted
            )
            mean = (hsum / wsafe[:, None]).to(hidden.dtype)
            pooled = torch.cat([pooled, mean], dim=-1)

        ones = torch.ones(points.shape[0], device=device)
        n_c = torch.zeros(n_cells, device=device).scatter_add_(0, cell, ones)
        mult = torch.log1p(n_c)
        total_q = torch.log1p(
            torch.zeros(n_cells, device=device).scatter_add_(0, cell, q_phys)
        )
        dtime = rel[:, 3].float()
        m1 = (
            torch.zeros(n_cells, device=device).scatter_add_(
                0, cell, weight * dtime
            )
            / wsafe
        )
        m2 = (
            torch.zeros(n_cells, device=device).scatter_add_(
                0, cell, weight * dtime * dtime
            )
            / wsafe
        )
        # Clamp INSIDE the square root: single-point cells have exactly zero
        # variance and radius, and the derivative of sqrt at 0 is infinite,
        # which would NaN the backward pass.
        dt_std = (m2 - m1 * m1).clamp(min=1e-12).sqrt()
        r2 = torch.zeros(n_cells, device=device).scatter_add_(
            0, cell, rel[:, :3].float().pow(2).sum(-1)
        )
        rms = (r2 / n_c.clamp(min=1)).clamp(min=1e-12).sqrt()
        extras = torch.stack([mult, total_q, dt_std, rms], dim=-1)

        return (
            torch.cat([pooled, extras.to(feat_dtype)], dim=-1),
            cents_raw,
        )

    def _pool_knn(
        self,
        padded_m: Tensor,
        valid: Tensor,
        points: Tensor,
        feats: Tensor,
        charge: Tensor,
        dest_pad: Tensor,
        starts_sub: Tensor,
        n_sub: int,
        n_max: int,
        feat_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Pool by a k-nearest-neighbour gather around each centroid.

        Uses the same relative encoding and masked max / charge-weighted
        mean as the Voronoi path, with the summary scalars computed over
        the neighbour sets so the second MLP sees a consistent layout in
        both modes.
        """
        device = points.device
        n_tokens = self.max_tokens
        token_dim = self.token_dim
        k_global = min(self.k_neighbors, n_max)

        fps_idx, knn_idx = farthest_point_sampling_with_knn(
            padded_m,
            valid,
            n_tokens,
            k_global,
            random_start=self.training,
            validate=False,
        )

        cent_flat = (starts_sub[:, None] + fps_idx).reshape(-1)
        cents_raw = points.index_select(0, cent_flat)

        # Padded [features | charge] and geometry for the neighbour gathers.
        feats_pad = torch.zeros(
            n_sub * n_max, token_dim + 1, device=device, dtype=feat_dtype
        )
        feats_pad.index_copy_(
            0,
            dest_pad,
            torch.cat([feats, charge.unsqueeze(-1).to(feat_dtype)], dim=-1),
        )
        points_pad = torch.zeros(
            n_sub * n_max, 4, device=device, dtype=points.dtype
        )
        points_pad.index_copy_(0, dest_pad, points)

        k_local = knn_idx.size(2)
        base = torch.arange(n_sub, device=device, dtype=knn_idx.dtype) * n_max
        flat_knn = (knn_idx + base.view(-1, 1, 1)).reshape(-1)

        neigh_aug = feats_pad.index_select(0, flat_knn).reshape(
            n_sub, n_tokens, k_local, token_dim + 1
        )
        neigh_feats = neigh_aug[..., :token_dim]
        knn_valid = (
            valid.reshape(-1)
            .index_select(0, flat_knn)
            .reshape(n_sub, n_tokens, k_local)
        )
        # The sampling backend pads a short neighbour list by repeating the
        # centroid index, which is itself a valid point, so `valid` alone
        # marks those slots as real. Padding always occupies the ranks at or
        # beyond the event's own point count, so mask those out as well --
        # otherwise the centroid is counted several times in the
        # multiplicity, total charge, mean and spread scalars. Reachable
        # whenever `k_neighbors` exceeds an event's point count, which on
        # this branch means `k_neighbors > max_tokens`.
        counts = valid.sum(dim=1)
        ranks = torch.arange(k_local, device=device)
        knn_valid = knn_valid & (ranks.view(1, 1, -1) < counts.view(-1, 1, 1))

        neigh_xyzt = points_pad.index_select(0, flat_knn).reshape(
            n_sub, n_tokens, k_local, 4
        )
        rel = neigh_xyzt - cents_raw.view(n_sub, n_tokens, 1, 4)
        neigh_feats = neigh_feats + self.rel_encoder(rel).to(neigh_feats.dtype)

        masked = neigh_feats.masked_fill(
            ~knn_valid.unsqueeze(-1), float("-inf")
        )
        pooled = masked.max(dim=2).values
        pooled = torch.where(
            torch.isfinite(pooled), pooled, torch.zeros_like(pooled)
        )

        knn_charge = neigh_aug[..., token_dim]
        if self.charge_weighted_mean:
            weight = (knn_charge * knn_valid.to(knn_charge.dtype)).unsqueeze(
                -1
            )
        else:
            weight = knn_valid.unsqueeze(-1).to(neigh_feats.dtype)
        if self.knn_pool == "max_mean":
            mean = (neigh_feats * weight).sum(dim=2) / weight.sum(dim=2).clamp(
                min=1e-6
            )
            pooled = torch.cat([pooled, mean.to(pooled.dtype)], dim=-1)

        # Summary scalars over the neighbour sets, in float32, matching the
        # Voronoi definitions.
        valid_f = knn_valid.float()
        n_c = valid_f.sum(dim=2)
        mult = torch.log1p(n_c)
        total_q = torch.log1p(
            (torch.expm1(knn_charge.float().clamp(min=0)) * valid_f).sum(dim=2)
        )
        wq = (
            (knn_charge.float() * valid_f)
            if self.charge_weighted_mean
            else valid_f
        )
        wsafe = wq.sum(dim=2).clamp(min=1e-6)
        dtime = rel[..., 3].float()
        m1 = (wq * dtime).sum(dim=2) / wsafe
        m2 = (wq * dtime * dtime).sum(dim=2) / wsafe
        # Clamp INSIDE the square root, see `_pool_voronoi`.
        dt_std = (m2 - m1 * m1).clamp(min=1e-12).sqrt()
        r2 = (rel[..., :3].float().pow(2).sum(-1) * valid_f).sum(dim=2)
        rms = (r2 / n_c.clamp(min=1)).clamp(min=1e-12).sqrt()
        extras = torch.stack([mult, total_q, dt_std, rms], dim=-1)

        pre_sub = torch.cat([pooled, extras.to(feat_dtype)], dim=-1).view(
            n_sub * n_tokens, -1
        )
        return pre_sub, cents_raw
