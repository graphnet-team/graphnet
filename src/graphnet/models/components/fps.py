"""Farthest point sampling (FPS) for point-cloud tokenization.

Ported from the Neptune reference implementation
(https://github.com/felixyu7/neptune, MIT licence), described in
https://arxiv.org/abs/2510.01733.

FPS greedily selects `K` points that are maximally spread out, which is how
`graphnet.models.components.tokenizers.FPSTokenizer` picks token centroids for
an event. Two fused variants are provided because the FPS loop already computes
every point-to-centroid distance: nearest-centroid (Voronoi) assignment and a
k-nearest-neighbour gather both come almost for free.

Two backends, selected automatically and producing identical indices:

- **Triton** (CUDA, non-float64), in
  :mod:`graphnet.models.components.fps_triton`. Streams distances, so nothing
  of size `[B, N, K]` is ever materialised.
- **Pure PyTorch**, below. Vectorised over the batch; the only Python loop is
  the inherently sequential `K` iterations. Note that the assignment path
  materialises a `[B, N, K]` distance tensor -- roughly 400 MB in float32 at
  `B=256, N=3000, K=128` -- which the Triton path avoids.

Semantics shared by both backends:

- validity = `valid_mask` AND all-finite coordinates,
- distances accumulate in float32 (float64 for double inputs),
- ties select the lowest index,
- an out-of-range start falls back to index 0; once valid candidates are
  exhausted (`K` > valid count) the last selection is repeated,
- kNN neighbours are closest-first over all valid points (the centroid and
  already-selected points included); rows with fewer than `k` valid points
  pad with the centroid index.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from graphnet.utilities.imports import has_triton_package

# ------------------------------------------------------------------------
# Pure-PyTorch backend
# ------------------------------------------------------------------------


def _acc(points: Tensor) -> Tensor:
    """Return `points` in the accumulation dtype (float32, or float64)."""
    return points if points.dtype == torch.float64 else points.float()


def _fps_state(
    points: Tensor, mask: Tensor, start_idx: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Set up the accumulators shared by the pure-PyTorch FPS routines."""
    _, n_points, _ = points.shape
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    inf = float("inf")
    min_d = torch.where(valid, inf, -inf).to(pts.dtype)
    in_range = (start_idx >= 0) & (start_idx < n_points)
    last = torch.where(in_range, start_idx, torch.zeros_like(start_idx))
    return pts, valid, min_d, last


def _fps_reference(
    points: Tensor, mask: Tensor, start_idx: Tensor, k: int
) -> Tensor:
    """Select `k` farthest points per row, returning `[B, k]` indices."""
    batch_size = points.shape[0]
    idx = torch.empty(batch_size, k, device=points.device, dtype=torch.long)
    pts, valid, min_d, last = _fps_state(points, mask, start_idx)
    rows = torch.arange(batch_size, device=points.device)
    for i in range(k):
        idx[:, i] = last
        min_d[rows, last] = -float("inf")
        if i + 1 == k:
            break
        centroid = pts[rows, last]
        dist = (pts - centroid[:, None, :]).square().sum(dim=2)
        # Kernel-exact update: invalid lanes keep -inf, and NaN distances
        # (only reachable via a pathological non-finite start) never
        # overwrite.
        min_d = torch.where(valid & (dist < min_d), dist, min_d)
        vals, nxt = min_d.max(dim=1)
        # Exhausted rows (all -inf) repeat the previous selection.
        last = torch.where(vals.isneginf(), last, nxt)
    return idx


def _assign_reference(points: Tensor, idx: Tensor) -> Tensor:
    """Assign every point to its nearest centroid among `idx`.

    Returns `[B, N]` int64 in `[0, K)`; values at invalid lanes are
    unspecified. Per-dimension accumulation in float32 matches the
    Triton kernels' order bitwise, and NaN distances promote to +inf so
    a degenerate centroid is never chosen.
    """
    batch_size, n_points, n_dims = points.shape
    pts = _acc(points)
    cents = torch.gather(
        pts, 1, idx.unsqueeze(-1).expand(-1, -1, n_dims)
    )  # [B, K, D]
    dist = pts.new_zeros(batch_size, n_points, idx.size(1))
    for dim in range(n_dims):
        diff = pts[:, :, dim].unsqueeze(-1) - cents[:, :, dim].unsqueeze(1)
        dist = dist + diff * diff
    dist = torch.where(dist == dist, dist, float("inf"))
    return dist.argmin(dim=-1)


def _fps_knn_reference(
    points: Tensor,
    mask: Tensor,
    start_idx: Tensor,
    k: int,
    k_neighbors: int,
) -> Tuple[Tensor, Tensor]:
    """Select `k` centroids and gather their `k_neighbors` nearest points."""
    _, _, n_dims = points.shape
    centroid_idx = _fps_reference(points, mask, start_idx, k)
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    cents = torch.gather(
        pts, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, n_dims)
    )
    dist = (cents.unsqueeze(2) - pts.unsqueeze(1)).square().sum(dim=-1)
    dist = torch.where(valid.unsqueeze(1), dist, float("inf"))
    # Stable sort => lowest index first among exact ties, like the kernels.
    svals, sidx = dist.sort(dim=-1, stable=True)
    svals, sidx = svals[..., :k_neighbors], sidx[..., :k_neighbors]
    # Slots past the row's valid count carry +inf -> pad with the centroid.
    return centroid_idx, torch.where(
        svals.isinf(), centroid_idx.unsqueeze(-1), sidx
    )


def _nearest_assign_reference(
    p_flat: Tensor,
    cents: Tensor,
    starts: Tensor,
    counts: Tensor,
    k: int,
) -> Tensor:
    """Nearest-centroid assignment over batch-segmented flat points."""
    n_points, n_dims = p_flat.shape
    batch_size = starts.numel()
    seg = torch.repeat_interleave(
        torch.arange(batch_size, device=p_flat.device),
        counts,
        output_size=n_points,
    )
    cents_per_point = cents.view(batch_size, k, n_dims).index_select(0, seg)
    pts = p_flat.float()
    dist = pts.new_zeros(n_points, k)
    for dim in range(n_dims):
        diff = pts[:, dim].unsqueeze(-1) - cents_per_point[:, :, dim]
        dist = dist + diff * diff
    dist = torch.where(dist == dist, dist, float("inf"))
    return dist.argmin(dim=1)


# ------------------------------------------------------------------------
# Backend dispatch
# ------------------------------------------------------------------------


def _use_triton(points: Tensor) -> bool:
    """Return whether the Triton backend can serve `points`."""
    return (
        points.is_cuda
        and points.dtype != torch.float64
        and has_triton_package()
    )


def _dispatch(
    points: Tensor,
    mask: Tensor,
    start_idx: Tensor,
    k: int,
    k_neighbors: Optional[int],
) -> Tensor | Tuple[Tensor, Tensor]:
    """Route a resolved call to the best available backend."""
    if _use_triton(points):
        # Imported here, never at module scope: `graphnet.models` is
        # walk-imported by `ModelConfig`'s class discovery, and Triton is
        # absent on e.g. macOS.
        from graphnet.models.components import fps_triton

        if k_neighbors is None:
            return fps_triton.fps(points, mask, start_idx, k)
        return fps_triton.fps_knn(points, mask, start_idx, k, k_neighbors)
    if k_neighbors is None:
        return _fps_reference(points, mask, start_idx, k)
    return _fps_knn_reference(points, mask, start_idx, k, k_neighbors)


# ------------------------------------------------------------------------
# Front-end: validation and start resolution
# ------------------------------------------------------------------------


def _resolve_start_idx(
    valid: Tensor,
    counts: Optional[Tensor],
    batch_size: int,
    n_points: int,
    k: int,
    device: torch.device,
    start_idx: Optional[Tensor],
    random_start: bool,
    generator: Optional[torch.Generator],
    validate: bool = True,
) -> Tensor:
    """Produce a validated contiguous `[B]` long start index.

    `valid` marks selectable points: mask-true AND all-finite coordinates.
    With `validate=False` the checks (and their host sync) are skipped
    entirely, and `counts` may then be None on the `start_idx is None` path.
    All start resolution is pure-tensor: no host-device sync beyond
    validation.
    """
    if start_idx is None:
        if validate:
            assert counts is not None
            if bool((counts < k).any()):
                raise ValueError(
                    "FPS requires K <= number of valid points. Found "
                    f"batch(es) with K={k} but fewer valid points."
                )
        if not random_start:
            # Deterministic start: first valid index per row (all-invalid
            # rows give 0; the backend pads those rows anyway).
            return valid.long().argmax(dim=1).contiguous()
        # Random start: masked-random argmax draws uniformly over valid
        # points without multinomial's float32 probability copy. All-invalid
        # rows argmax to 0, matching the documented pad behaviour.
        scores = torch.rand(
            batch_size, n_points, device=device, generator=generator
        )
        scores = torch.where(valid, scores, float("-inf"))
        return scores.argmax(dim=1).contiguous()

    # User-supplied path. Fuse all validation into a single sync. Bad inputs
    # (K > counts, out-of-range) raise; a start index pointing at an invalid
    # slot is silently repaired to the first valid index in that row.
    if start_idx.device != device:
        raise ValueError(
            "start_idx must be on the same device as points (a cross-device "
            "copy here would silently synchronize the host)"
        )
    start_idx = start_idx.to(dtype=torch.long)
    if start_idx.numel() != batch_size:
        raise ValueError("start_idx must have shape [B]")
    start_idx = start_idx.reshape(batch_size)

    if validate:
        assert counts is not None
        out_of_range = (start_idx < 0) | (start_idx >= n_points)
        insufficient = counts < k
        if bool((out_of_range | insufficient).any()):
            if bool(insufficient.any()):
                raise ValueError(
                    "FPS requires K <= number of valid points. Found "
                    f"batch(es) with K={k} but fewer valid points."
                )
            raise ValueError("start_idx values must be within [0, N)")

    # Repair a start index pointing at an invalid slot. Pure-tensor, no sync.
    assert counts is not None
    has_valid = counts > 0
    safe_start = start_idx.clamp(0, max(n_points - 1, 0))
    supplied_valid = valid.gather(1, safe_start.unsqueeze(-1)).squeeze(-1)
    first_valid = torch.argmax(valid.long(), dim=1)
    repaired = torch.where(supplied_valid | ~has_valid, start_idx, first_valid)
    return repaired.contiguous()


def _prepare(
    points: Tensor, valid_mask: Tensor, precision: Optional[torch.dtype]
) -> Tuple[Tensor, Tensor]:
    """Cast inputs to the compute dtype and make them contiguous."""
    device = points.device
    if precision is None:
        if points.dtype != torch.float32:
            points = points.to(dtype=torch.float32)
    else:
        if device.type == "cpu" and precision == torch.bfloat16:
            raise ValueError(
                "bfloat16 is not supported on CPU (use float16, float32, or "
                "float64)"
            )
        if points.dtype != precision:
            points = points.to(dtype=precision)
    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)
    return points.contiguous(), valid_mask.contiguous()


def _prepare_and_resolve(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    start_idx: Optional[Tensor],
    random_start: bool,
    generator: Optional[torch.Generator],
    precision: Optional[torch.dtype],
    validate: bool,
    assume_finite: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run shape checks, dtype prep, validity and start resolution."""
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if points.shape[:2] != valid_mask.shape:
        raise ValueError(
            "points and valid_mask must agree on batch & point dims"
        )
    if k < 0:
        raise ValueError("K must be non-negative")

    device = points.device
    points_c, mask_c = _prepare(points, valid_mask, precision)
    batch_size, n_points, _ = points_c.shape

    # Empty-input edge cases: with K == 0 skip start resolution entirely (the
    # callers early-return an empty result and never use start_idx; argmax
    # over N == 0 would raise). N == 0 with K > 0 can never satisfy
    # K <= valid points, so reject it deterministically even when
    # validate=False.
    if k == 0:
        return (
            points_c,
            mask_c,
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )
    if n_points == 0:
        raise ValueError(
            "FPS with K > 0 requires at least one point (got N=0)"
        )

    # Selectable = mask-true AND all-finite; validation and start repair both
    # count from this same predicate, so validate=True enforces exactly the
    # documented "K <= valid points" precondition. assume_finite skips the
    # [B, N, D] finiteness pass on the caller's guarantee.
    valid = (
        mask_c if assume_finite else (mask_c & points_c.isfinite().all(dim=-1))
    )
    counts = (
        valid.sum(dim=1, dtype=torch.long)
        if (validate or start_idx is not None)
        else None
    )
    resolved = _resolve_start_idx(
        valid,
        counts,
        batch_size,
        n_points,
        k,
        device,
        start_idx,
        random_start,
        generator,
        validate,
    )
    return points_c, mask_c, resolved


# ------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tensor:
    """Select `k` maximally spread-out points per batch row.

    Args:
        points: Float tensor `[B, N, D]` (batch, points, coordinates).
        valid_mask: Bool tensor `[B, N]`; False marks padded/invalid points.
        k: Number of samples to draw per batch element. Must satisfy
            `k <= number of valid points` for every row.
        start_idx: Optional `[B]` long tensor giving the first index per row.
            Must live on the same device as `points`; an index pointing at an
            invalid slot is repaired to that row's first valid index.
        random_start: If True (default) and `start_idx` is not given, draw a
            random first index from the valid points. If False, the row's
            first valid index is used. Callers typically pass `self.training`
            so that the random start acts as a train-time augmentation and
            evaluation stays deterministic.
        generator: Optional generator for deterministic random starts.
        precision: Optional dtype for internal computation. None (default)
            uses float32 on all devices. float16/float32/float64 everywhere,
            bfloat16 on GPU only.
        validate: If True (default), verify `k <= valid count` per row and
            range-check a user-supplied `start_idx`. Costs one host-device
            sync; callers that already guarantee the precondition can pass
            False to keep the call fully asynchronous. With `validate=False`
            a violated precondition is not diagnosed -- the output is padded
            with repeated indices instead.
        assume_finite: If True, the caller guarantees every mask-true point
            has finite coordinates and the `[B, N, D]` finiteness pass is
            skipped.

    Returns:
        Long tensor `[B, k]` of selected point indices.
    """
    points_c, mask_c, resolved = _prepare_and_resolve(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
        assume_finite,
    )
    if k == 0:
        return torch.zeros(
            (points_c.shape[0], 0),
            device=points_c.device,
            dtype=torch.long,
        )
    result = _dispatch(points_c, mask_c, resolved, k, None)
    assert isinstance(result, Tensor)
    return result


def farthest_point_sampling_with_knn(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    k_neighbors: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run FPS and gather each centroid's `k_neighbors` nearest points.

    The distances computed during FPS are reused for the neighbour search.
    Neighbours are sorted closest-first; the centroid itself and
    already-selected points are eligible, and rows with fewer than
    `k_neighbors` valid points pad with the centroid index. All other
    arguments behave as in :func:`farthest_point_sampling`.

    Args:
        points: Float tensor `[B, N, D]`.
        valid_mask: Bool tensor `[B, N]`; False marks padded/invalid points.
        k: Number of centroids to draw per batch element.
        k_neighbors: Neighbours per centroid; must satisfy `0 < k_neighbors
            <= N`.
        start_idx: Optional `[B]` long tensor giving the first index per row.
        random_start: Draw a random first index when `start_idx` is None.
        generator: Optional generator for deterministic random starts.
        precision: Optional dtype for internal computation.
        validate: Verify the `k <= valid count` precondition.
        assume_finite: Skip the finiteness pass over `points`.

    Returns:
        Tuple of the `[B, k]` centroid indices and the `[B, k, k_neighbors]`
        neighbour indices, sorted by distance (closest first).
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    points_c, mask_c, resolved = _prepare_and_resolve(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
        assume_finite,
    )
    batch_size, n_points, _ = points_c.shape

    if k == 0:
        return (
            torch.zeros(
                (batch_size, 0), device=points_c.device, dtype=torch.long
            ),
            torch.zeros(
                (batch_size, 0, k_neighbors),
                device=points_c.device,
                dtype=torch.long,
            ),
        )

    if k_neighbors > n_points:
        raise ValueError(
            f"k_neighbors ({k_neighbors}) must be <= N ({n_points})"
        )

    result = _dispatch(points_c, mask_c, resolved, k, k_neighbors)
    assert isinstance(result, tuple)
    return result


def farthest_point_sampling_with_assign(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run FPS and assign every point to its nearest centroid.

    The FPS loop already computes each point's distance to every selected
    centroid, so the Voronoi assignment is nearly free on the fused CUDA
    path. Ties assign to the lowest centroid index. All other arguments
    behave as in :func:`farthest_point_sampling`.

    Args:
        points: Float tensor `[B, N, D]`.
        valid_mask: Bool tensor `[B, N]`; False marks padded/invalid points.
        k: Number of centroids to draw per batch element.
        start_idx: Optional `[B]` long tensor giving the first index per row.
        random_start: Draw a random first index when `start_idx` is None.
        generator: Optional generator for deterministic random starts.
        precision: Optional dtype for internal computation.
        validate: Verify the `k <= valid count` precondition.
        assume_finite: Skip the finiteness pass over `points`.

    Returns:
        Tuple of the `[B, k]` centroid indices and a `[B, N]` assignment
        tensor holding each point's nearest centroid as a position in
        `[0, k)` -- an index into the centroid tensor, not into `points`.
        Values at invalid lanes are unspecified.
    """
    points_c, mask_c, resolved = _prepare_and_resolve(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
        assume_finite,
    )
    batch_size, n_points, n_dims = points_c.shape

    if k == 0:
        return (
            torch.zeros(
                (batch_size, 0), device=points_c.device, dtype=torch.long
            ),
            torch.zeros(
                (batch_size, n_points),
                device=points_c.device,
                dtype=torch.long,
            ),
        )

    if _use_triton(points_c):
        from graphnet.models.components import fps_triton

        cap = fps_triton._single_tile_cap(fps_triton.SINGLE_TILE_MAX_N, n_dims)
        if n_points <= cap:
            return fps_triton.fps_assign(points_c, mask_c, resolved, k)

    # Composed route: tiled-N CUDA, CPU, and other devices.
    idx = _dispatch(points_c, mask_c, resolved, k, None)
    assert isinstance(idx, Tensor)
    return idx, _assign_reference(points_c, idx)


def nearest_assign(
    p_flat: Tensor,
    cents: Tensor,
    starts: Tensor,
    counts: Tensor,
    n_max: int,
    k: int,
) -> Tensor:
    """Assign batch-segmented flat points to their nearest centroid.

    Used by the charge-weighted Lloyd refinement in
    :class:`~graphnet.models.components.tokenizers.FPSTokenizer`, where the
    centroids are updated cell means rather than points of the cloud.
    Deterministic (lowest centroid index on exact ties) and gradient-free:
    the inputs are detached, since assignment is integer routing.

    Args:
        p_flat: Float32 `[N, D]` flat points, rows grouped per event.
        cents: Float32 `[B * k, D]` centroids.
        starts: Int64 `[B]` start offset of each event in `p_flat`.
        counts: Int64 `[B]` number of points in each event.
        n_max: Largest value in `counts`, as a host-side int.
        k: Number of centroids per event.

    Returns:
        Int64 `[N]` cell indices in `[0, k)`.
    """
    p_detached = p_flat.detach()
    c_detached = cents.detach()
    if p_detached.numel() == 0 or starts.numel() == 0:
        return torch.empty(
            p_detached.shape[0], device=p_detached.device, dtype=torch.long
        )
    if _use_triton(p_detached):
        from graphnet.models.components import fps_triton

        return fps_triton.nearest_assign(
            p_detached, c_detached, starts, counts, n_max, k
        )
    return _nearest_assign_reference(p_detached, c_detached, starts, counts, k)
