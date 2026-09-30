"""Learnable tokenizers turning point clouds into transformer tokens."""

from typing import Callable, List, Optional, Tuple

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
    (https://github.com/felixyu7/neptune, MIT licence).

    A per-point MLP lifts the features. Events with at most `max_tokens`
    points keep one token per point; larger events are reduced to
    `max_tokens` centroids by FPS in `(x, y, z, t)` and pooled per token.
    Four summary scalars (multiplicity, total charge, time spread, RMS
    radius) are appended and a second MLP refines every token.
    """

    N_EXTRA = 4

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
        metric_time_scale: float = 18.0,
        lloyd_iters: int = 0,
    ):
        """Construct `FPSTokenizer`.

        Args:
            feature_dim: Number of input features per point.
            max_tokens: Maximum number of tokens per event.
            token_dim: Output token dimension.
            mlp_layers: Hidden sizes of the per-point MLP; defaults to
                `[256, 512, 768]`.
            k_neighbors: Neighbours per centroid when `assign_mode="knn"`.
            dropout: Dropout inside both MLPs.
            knn_pool: `"max"`, or `"max_mean"` to also concatenate a weighted
                mean.
            rel_pos_hidden: Hidden size of the relative-offset encoder.
            charge_weighted_mean: Weight means and spreads by `log1p` charge.
            charge_col: Feature column holding `log1p` charge.
            assign_mode: `"voronoi"` pools each point into its nearest
                centroid; `"knn"` pools the `k_neighbors` nearest points of
                each centroid.
            metric_time_scale: Weight of time relative to space in the FPS
                and assignment metric only. The default, 18, is 0.3 km/us in
                `IceCube86` units.
            lloyd_iters: Charge-weighted Lloyd (k-means) steps refining the
                Voronoi centroids, which then become cell means; cells can
                end up empty. Ignored for `"knn"`.
        """
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 512, 768]
        if assign_mode not in ("voronoi", "knn"):
            raise ValueError(f"Unknown assign_mode '{assign_mode}'")
        if lloyd_iters < 0:
            raise ValueError(f"lloyd_iters must be >= 0, got {lloyd_iters}")
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.knn_pool = knn_pool
        self.charge_weighted_mean = charge_weighted_mean
        self.charge_col = charge_col
        self.assign_mode = assign_mode
        self.lloyd_iters = lloyd_iters

        mlp1: List[nn.Module] = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp1 += [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp1 = nn.Sequential(*mlp1, nn.Linear(in_dim, token_dim))

        self.pooled_dim = token_dim * (2 if knn_pool == "max_mean" else 1)
        self.mlp2 = nn.Sequential(
            nn.Linear(self.pooled_dim + self.N_EXTRA, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        # Embeds each point's (dx, dy, dz, dt) offset from its centroid.
        self.rel_encoder = nn.Sequential(
            nn.Linear(4, rel_pos_hidden),
            nn.GELU(),
            nn.Linear(rel_pos_hidden, token_dim),
        )
        self.register_buffer(
            "metric_scale",
            torch.tensor([1.0, 1.0, 1.0, float(metric_time_scale)]),
            persistent=False,
        )

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
            coords: `[N, 3]` positions `(x, y, z)`.
            features: `[N, F]` per-point features.
            batch_ids: `[N]` event index of each point.
            times: `[N, 1]` time of each point.
            batch_size: Number of events, so that trailing empty events keep
                their row; inferred from `batch_ids` if None.

        Returns:
            `(tokens [B, max_tokens, token_dim], centroids [B, max_tokens, 4],
            masks [B, max_tokens])`.
        """
        device, dtype_p = coords.device, coords.dtype
        n_tokens = self.max_tokens
        if coords.numel() == 0:  # reachable: every event has zero pulses
            n_events = batch_size or 0
            return (
                features.new_zeros(n_events, n_tokens, self.token_dim),
                coords.new_zeros(n_events, n_tokens, 4),
                torch.zeros(n_events, n_tokens, device=device).bool(),
            )

        batch_idx = batch_ids.long()
        points = torch.cat([coords[:, :3], times], dim=-1)
        point_feats = self.mlp1(features)
        feat_dtype = point_feats.dtype  # follows autocast

        # The one host sync: counting on a CPU copy of the batch indices.
        counts = torch.bincount(
            batch_idx.cpu(), minlength=batch_size or 0
        ).tolist()
        n_events = len(counts)
        sort_idx = torch.argsort(batch_idx, stable=True)
        p_sorted = points.index_select(0, sort_idx)
        f_sorted = point_feats.index_select(0, sort_idx)
        q_sorted = features[:, self.charge_col].index_select(0, sort_idx)

        offsets = [0] * n_events
        for i in range(1, n_events):
            offsets[i] = offsets[i - 1] + counts[i - 1]
        small = [i for i, c in enumerate(counts) if c <= n_tokens]
        large = [i for i, c in enumerate(counts) if c > n_tokens]
        # Routing (rows, counts, starts) in one non-blocking host-to-device
        # copy; small events first.
        rows = small + large
        routing = torch.tensor(
            [rows, [counts[i] for i in rows], [offsets[i] for i in rows]],
            dtype=torch.long,
        )
        if device.type == "cuda":
            routing = routing.pin_memory()
        routing = routing.to(device, non_blocking=True)
        n_small = len(small)

        pre = torch.zeros(
            n_events,
            n_tokens,
            self.pooled_dim + self.N_EXTRA,
            device=device,
            dtype=feat_dtype,
        )
        cents = torch.zeros(
            n_events, n_tokens, 4, device=device, dtype=dtype_p
        )
        masks = torch.zeros(n_events, n_tokens, device=device).bool()
        sorted_inputs = (f_sorted, p_sorted, q_sorted)
        if small:
            self._tokenize_small(
                pre,
                cents,
                masks,
                routing[:, :n_small],
                sum(counts[i] for i in small),
                *sorted_inputs,
            )
        if large:
            self._tokenize_large(
                pre,
                cents,
                masks,
                routing[:, n_small:],
                sum(counts[i] for i in large),
                max(counts[i] for i in large),
                *sorted_inputs,
            )
        mask = masks.unsqueeze(-1)
        return (self.mlp2(pre) * mask).to(features.dtype), cents * mask, masks

    @staticmethod
    def _route(routing: Tensor, n_pts: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Map a `[3, B_sub]` (rows, counts, starts) routing to point indices.

        Returns:
            `(seg, local, src)`: each point's subset event, its position in
            that event and its row in the batch-sorted arrays.
        """
        _, counts, starts = routing
        seg = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device),
            counts,
            output_size=n_pts,
        )
        local = (
            torch.arange(n_pts, device=counts.device)
            - (torch.cumsum(counts, 0) - counts)[seg]
        )
        return seg, local, starts[seg] + local

    def _tokenize_small(
        self,
        pre: Tensor,
        cents: Tensor,
        masks: Tensor,
        routing: Tensor,
        n_pts: int,
        f_sorted: Tensor,
        p_sorted: Tensor,
        q_sorted: Tensor,
    ) -> None:
        """Fill the outputs for events with one token per point."""
        rows, counts = routing[0], routing[1]
        n_sub, n_tokens = rows.numel(), self.max_tokens
        seg, local, src = self._route(routing, n_pts)
        feats = f_sorted.index_select(0, src)
        charge = q_sorted.index_select(0, src)
        dest = seg * n_tokens + local
        if self.knn_pool == "max_mean":
            feats = torch.cat([feats, feats], dim=-1)
        # One point: log1p(1) multiplicity, its own charge, zero spreads.
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
        pre_sub = pre.new_zeros(n_sub * n_tokens, pre.shape[-1])
        pre_sub.index_copy_(
            0, dest, torch.cat([feats, extras.to(pre.dtype)], -1)
        )
        cents_sub = cents.new_zeros(n_sub * n_tokens, 4)
        cents_sub.index_copy_(0, dest, p_sorted.index_select(0, src))
        pre.index_copy_(0, rows, pre_sub.view(n_sub, n_tokens, -1))
        cents.index_copy_(0, rows, cents_sub.view(n_sub, n_tokens, 4))
        slots = torch.arange(n_tokens, device=pre.device)
        masks.index_copy_(0, rows, slots[None, :] < counts[:, None])

    def _tokenize_large(
        self,
        pre: Tensor,
        cents: Tensor,
        masks: Tensor,
        routing: Tensor,
        n_pts: int,
        n_max: int,
        f_sorted: Tensor,
        p_sorted: Tensor,
        q_sorted: Tensor,
    ) -> None:
        """Fill the outputs for events reduced by FPS and pooling."""
        rows, counts = routing[0], routing[1]
        n_sub, n_tokens = rows.numel(), self.max_tokens
        seg, local, src = self._route(routing, n_pts)
        feats = f_sorted.index_select(0, src)
        points = p_sorted.index_select(0, src)
        charge = q_sorted.index_select(0, src)

        # Float32 copy in the sampling metric, padded to [n_sub, n_max, 4].
        points_m = points.float() * self.metric_scale
        dest_pad = seg * n_max + local
        padded_m = points_m.new_zeros(n_sub * n_max, 4)
        padded_m.index_copy_(0, dest_pad, points_m)
        padded_m = padded_m.view(n_sub, n_max, 4)
        valid = (
            torch.arange(n_max, device=pre.device)[None, :] < counts[:, None]
        )
        starts = torch.cumsum(counts, 0) - counts  # within the subset

        pool = self._pool_knn
        if self.assign_mode == "voronoi":
            pool = self._pool_voronoi
        pre_sub, cents_sub = pool(
            padded_m,
            valid,
            points,
            points_m,
            feats,
            charge,
            seg,
            dest_pad,
            starts,
            counts,
        )
        pre.index_copy_(0, rows, pre_sub.view(n_sub, n_tokens, -1))
        cents.index_copy_(
            0, rows, cents_sub.view(n_sub, n_tokens, 4).to(cents.dtype)
        )
        masks.index_copy_(0, rows, torch.ones_like(masks[:n_sub]))

    def _summary(
        self,
        cell_sum: Callable[[Tensor], Tensor],
        charge: Tensor,
        weight: Tensor,
        rel: Tensor,
    ) -> Tensor:
        """Per-token multiplicity, total charge, time spread and RMS radius.

        Args:
            cell_sum: Sums per-point values into per-token totals.
            charge: Per-point `log1p` charge.
            weight: Per-point weight of the time moments.
            rel: Per-point `(dx, dy, dz, dt)` offset from its centroid.
        """
        n_c = cell_sum(torch.ones_like(weight))
        total_q = cell_sum(torch.expm1(charge.float().clamp(min=0)))
        wsafe = cell_sum(weight).clamp(min=1e-6)
        dtime = rel[..., 3].float()
        m1 = cell_sum(weight * dtime) / wsafe
        m2 = cell_sum(weight * dtime * dtime) / wsafe
        # Clamp inside the sqrt: its derivative at zero spread is infinite.
        dt_std = (m2 - m1 * m1).clamp(min=1e-12).sqrt()
        r2 = cell_sum(rel[..., :3].float().pow(2).sum(-1))
        rms = (r2 / n_c.clamp(min=1)).clamp(min=1e-12).sqrt()
        return torch.stack(
            [torch.log1p(n_c), torch.log1p(total_q), dt_std, rms], dim=-1
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
        starts: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Pool every point into its nearest centroid."""
        device, n_tokens, token_dim = (
            points.device,
            self.max_tokens,
            self.token_dim,
        )
        n_sub, n_max = padded_m.shape[:2]
        n_cells = n_sub * n_tokens
        # `validate=False` is safe: every event here has > n_tokens points.
        fps_idx, assign_pad = farthest_point_sampling_with_assign(
            padded_m,
            valid,
            n_tokens,
            random_start=self.training,
            validate=False,
            assume_finite=True,
        )
        cent_flat = (starts[:, None] + fps_idx).reshape(-1)
        cents = points.index_select(0, cent_flat)
        cell = seg * n_tokens + assign_pad.reshape(-1)[dest_pad]

        def cell_sum(values: Tensor) -> Tensor:
            out = torch.zeros(n_cells, *values.shape[1:], device=device)
            index = cell.view(-1, *[1] * (values.dim() - 1)).expand_as(values)
            return out.scatter_add_(0, index, values)

        if self.lloyd_iters > 0:
            q_phys = torch.expm1(charge.float().clamp(min=0))[:, None]
            cm = points_m.index_select(0, cent_flat)
            for _ in range(self.lloyd_iters):
                wsum = cell_sum(q_phys[:, 0])[:, None]
                mean = cell_sum(points_m * q_phys) / wsum.clamp(min=1e-9)
                cm = torch.where(wsum > 0, mean, cm)  # empty cells stay put
                assign = nearest_assign(
                    points_m, cm, starts, counts, n_max, n_tokens
                )
                cell = seg * n_tokens + assign
            # Centroids become charge-weighted means of the final cells.
            wsum = cell_sum(q_phys[:, 0])[:, None]
            mean = cell_sum(points * q_phys) / wsum.clamp(min=1e-9)
            cents = torch.where(wsum > 0, mean.to(cents.dtype), cents)

        rel = points - cents.index_select(0, cell)
        hidden = feats + self.rel_encoder(rel).to(feats.dtype)
        cell_t = cell.unsqueeze(-1).expand(-1, token_dim)
        pooled = torch.full(
            (n_cells, token_dim),
            float("-inf"),
            device=device,
            dtype=hidden.dtype,
        ).scatter_reduce(0, cell_t, hidden, reduce="amax", include_self=True)
        pooled = torch.where(pooled.isfinite(), pooled, 0.0)

        weight = charge.float()
        if not self.charge_weighted_mean:
            weight = torch.ones_like(weight)
        if self.knn_pool == "max_mean":
            hsum = cell_sum(hidden.float() * weight[:, None])
            wsafe = cell_sum(weight).clamp(min=1e-6)[:, None]
            pooled = torch.cat([pooled, (hsum / wsafe).to(hidden.dtype)], -1)
        extras = self._summary(cell_sum, charge, weight, rel)
        return torch.cat([pooled, extras.to(pooled.dtype)], dim=-1), cents

    def _pool_knn(
        self,
        padded_m: Tensor,
        valid: Tensor,
        points: Tensor,
        points_m: Tensor,
        feats: Tensor,
        charge: Tensor,
        seg: Tensor,
        dest_pad: Tensor,
        starts: Tensor,
        counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Pool the `k_neighbors` nearest points of every centroid."""
        device, n_tokens, token_dim = (
            points.device,
            self.max_tokens,
            self.token_dim,
        )
        n_sub, n_max = padded_m.shape[:2]
        fps_idx, knn_idx = farthest_point_sampling_with_knn(
            padded_m,
            valid,
            n_tokens,
            min(self.k_neighbors, n_max),
            random_start=self.training,
            validate=False,
        )
        cents = points.index_select(0, (starts[:, None] + fps_idx).reshape(-1))

        # Padded [features | charge] and positions, gathered per neighbour.
        aug = torch.cat([feats, charge.unsqueeze(-1).to(feats.dtype)], dim=-1)
        aug_pad = aug.new_zeros(n_sub * n_max, token_dim + 1)
        aug_pad.index_copy_(0, dest_pad, aug)
        points_pad = points.new_zeros(n_sub * n_max, 4)
        points_pad.index_copy_(0, dest_pad, points)
        k = knn_idx.size(2)
        flat = (
            knn_idx + torch.arange(n_sub, device=device).view(-1, 1, 1) * n_max
        )
        flat = flat.reshape(-1)
        shape = (n_sub, n_tokens, k)
        neigh = aug_pad.index_select(0, flat).reshape(*shape, token_dim + 1)
        neigh_feats, knn_charge = neigh[..., :token_dim], neigh[..., token_dim]
        # Short rows are padded with the centroid index: mask ranks beyond
        # the event's point count so it is not counted repeatedly.
        knn_valid = valid.reshape(-1).index_select(0, flat).reshape(shape)
        ranks = torch.arange(k, device=device).view(1, 1, -1)
        knn_valid = knn_valid & (ranks < counts.view(-1, 1, 1))

        rel = points_pad.index_select(0, flat).reshape(*shape, 4)
        rel = rel - cents.view(n_sub, n_tokens, 1, 4)
        neigh_feats = neigh_feats + self.rel_encoder(rel).to(neigh_feats.dtype)
        pooled = neigh_feats.masked_fill(
            ~knn_valid.unsqueeze(-1), float("-inf")
        )
        pooled = pooled.max(dim=2).values
        pooled = torch.where(pooled.isfinite(), pooled, 0.0)

        if self.knn_pool == "max_mean":
            weight = knn_valid.unsqueeze(-1).to(neigh_feats.dtype)
            if self.charge_weighted_mean:
                weight = (knn_charge * knn_valid.to(knn_charge.dtype))[
                    ..., None
                ]
            mean = (neigh_feats * weight).sum(dim=2)
            mean = mean / weight.sum(dim=2).clamp(min=1e-6)
            pooled = torch.cat([pooled, mean.to(pooled.dtype)], dim=-1)

        valid_f = knn_valid.float()
        weight = knn_charge.float()
        if not self.charge_weighted_mean:
            weight = torch.ones_like(weight)
        extras = self._summary(
            lambda values: (values * valid_f).sum(dim=2),
            knn_charge,
            weight,
            rel,
        )
        pre_sub = torch.cat([pooled, extras.to(pooled.dtype)], dim=-1)
        return pre_sub.view(n_sub * n_tokens, -1), cents
