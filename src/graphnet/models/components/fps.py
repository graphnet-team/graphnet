"""Farthest point sampling (FPS) for point-cloud tokenization.

Ported from the Neptune reference implementation
(https://github.com/felixyu7/neptune, MIT licence). Calls go to the optional
`torch_fps` package (Triton kernels on CUDA, C++ on CPU) when it is installed;
otherwise the pure-PyTorch implementation below is used.

Shared semantics: a point is valid if its mask is True and its coordinates are
finite; distances accumulate in float32 (or `precision=torch.float64`); ties
select the lowest index; once a row runs out of valid points the last
selection repeats; kNN rows with fewer than `k_neighbors` valid points pad
with the centroid index. The backends agree up to float32 rounding, which can
flip exact ties.
"""

import functools
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import torch
from torch import Tensor

from graphnet.utilities.imports import has_torch_fps_package

F = TypeVar("F", bound=Callable[..., Any])


def _torch_fps() -> Any:
    """Return the `torch_fps` module if it is installed, else None.

    `torch_fps` picks its own backend: Triton on CUDA, C++ on CPU.
    """
    if has_torch_fps_package():
        import torch_fps  # pyright: reportMissingImports=false

        return torch_fps
    return None


def _dispatch(function: F) -> F:
    """Route calls to the same-named `torch_fps` function when possible."""

    @functools.wraps(function)
    def wrapper(points: Tensor, *args: Any, **kwargs: Any) -> Any:
        backend = _torch_fps()
        if backend is not None:
            fast = getattr(backend, function.__name__)
            return fast(points, *args, **kwargs)
        return function(points, *args, **kwargs)

    return cast(F, wrapper)


def _acc(points: Tensor) -> Tensor:
    """Return `points` in the accumulation dtype (float32, or float64)."""
    return points if points.dtype == torch.float64 else points.float()


def _prepare(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    start_idx: Optional[Tensor],
    random_start: bool,
    generator: Optional[torch.Generator],
    precision: Optional[torch.dtype],
    validate: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Check shapes and return `(points, valid, start)` for the reference."""
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.shape != points.shape[:2]:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if k < 0:
        raise ValueError("K must be non-negative")
    batch_size, n_points, _ = points.shape
    if k > 0 and n_points == 0:
        raise ValueError(
            "FPS with K > 0 requires at least one point (got N=0)"
        )
    pts = _acc(points.to(precision or torch.float32))
    valid = valid_mask.to(points.device, torch.bool) & pts.isfinite().all(-1)
    if k == 0:
        return pts, valid, valid.new_zeros(batch_size, dtype=torch.long)

    if validate:
        if bool((valid.sum(dim=1) < k).any()):
            raise ValueError(
                "FPS requires K <= number of valid points. Found batch(es) "
                f"with K={k} but fewer valid points."
            )
        if start_idx is not None and bool(
            ((start_idx < 0) | (start_idx >= n_points)).any()
        ):
            raise ValueError("start_idx values must be within [0, N)")

    first_valid = valid.long().argmax(dim=1)
    if start_idx is None:
        if not random_start:
            return pts, valid, first_valid
        # Uniform over valid points: argmax of masked random scores.
        scores = torch.rand(
            batch_size, n_points, device=points.device, generator=generator
        )
        start = torch.where(valid, scores, -1.0).argmax(dim=1)
        return pts, valid, start

    # A start at an invalid slot is repaired to the row's first valid point.
    start = start_idx.long().reshape(batch_size)
    safe = start.clamp(0, n_points - 1)
    ok = valid.gather(1, safe.unsqueeze(-1)).squeeze(-1) | ~valid.any(dim=1)
    return pts, valid, torch.where(ok, start, first_valid)


def _fps_reference(
    pts: Tensor, valid: Tensor, start_idx: Tensor, k: int
) -> Tensor:
    """Select `k` farthest points per row, returning `[B, k]` indices."""
    batch_size, n_points, _ = pts.shape
    idx = torch.empty(batch_size, k, device=pts.device, dtype=torch.long)
    min_d = torch.where(valid, float("inf"), -float("inf")).to(pts.dtype)
    in_range = (start_idx >= 0) & (start_idx < n_points)
    last = torch.where(in_range, start_idx, torch.zeros_like(start_idx))
    rows = torch.arange(batch_size, device=pts.device)
    for i in range(k):
        idx[:, i] = last
        min_d[rows, last] = -float("inf")
        if i + 1 == k:
            break
        dist = (pts - pts[rows, last][:, None, :]).square().sum(dim=2)
        # Invalid lanes keep -inf; NaN distances never overwrite.
        min_d = torch.where(valid & (dist < min_d), dist, min_d)
        vals, nxt = min_d.max(dim=1)
        last = torch.where(vals.isneginf(), last, nxt)  # exhausted: repeat
    return idx


def _sq_dist(points: Tensor, cents: Tensor) -> Tensor:
    """Return `[..., N, K]` squared distances, NaN promoted to +inf."""
    dist = points.new_zeros(*points.shape[:-1], cents.shape[-2])
    for dim in range(points.shape[-1]):
        diff = points[..., :, dim, None] - cents[..., None, :, dim]
        dist = dist + diff * diff
    return torch.where(dist == dist, dist, float("inf"))


def _gather_rows(pts: Tensor, idx: Tensor) -> Tensor:
    """Gather `pts[b, idx[b]]` as `[B, K, D]`."""
    return torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, pts.size(-1)))


@_dispatch
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
        points: Float `[B, N, D]` coordinates.
        valid_mask: Bool `[B, N]`; False marks padded or invalid points.
        k: Number of points to select per row; at most the valid count.
        start_idx: Optional `[B]` first index per row, on `points.device`.
        random_start: Draw a random valid start when `start_idx` is None;
            otherwise start at the first valid point.
        generator: Optional generator for the random start.
        precision: Compute dtype; float32 by default, float64 on request.
        validate: Check `k <= valid count` and `start_idx` (one host sync).
        assume_finite: Caller guarantees finite coordinates for valid points.

    Returns:
        Long `[B, k]` selected indices.
    """
    pts, valid, start = _prepare(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
    )
    return _fps_reference(pts, valid, start, k)


@_dispatch
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

    Arguments are as in :func:`farthest_point_sampling`, plus `k_neighbors`
    (`0 < k_neighbors <= N`).

    Returns:
        `[B, k]` centroid indices and `[B, k, k_neighbors]` neighbour indices,
        closest first.
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if points.dim() == 3 and k > 0 and k_neighbors > points.shape[1]:
        raise ValueError(
            f"k_neighbors ({k_neighbors}) must be <= N ({points.shape[1]})"
        )
    pts, valid, start = _prepare(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
    )
    idx = _fps_reference(pts, valid, start, k)
    dist = _sq_dist(_gather_rows(pts, idx), pts)
    dist = torch.where(valid.unsqueeze(1), dist, float("inf"))
    svals, sidx = dist.sort(dim=-1, stable=True)  # stable: lowest index first
    svals, sidx = svals[..., :k_neighbors], sidx[..., :k_neighbors]
    return idx, torch.where(svals.isinf(), idx.unsqueeze(-1), sidx)


@_dispatch
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

    Arguments are as in :func:`farthest_point_sampling`.

    Returns:
        `[B, k]` centroid indices and `[B, N]` assignments in `[0, k)`
        (unspecified at invalid points).
    """
    pts, valid, start = _prepare(
        points,
        valid_mask,
        k,
        start_idx,
        random_start,
        generator,
        precision,
        validate,
    )
    idx = _fps_reference(pts, valid, start, k)
    if k == 0:
        return idx, torch.zeros_like(valid, dtype=torch.long)
    return idx, _sq_dist(pts, _gather_rows(pts, idx)).argmin(dim=-1)


@_dispatch
def nearest_assign(
    p_flat: Tensor,
    cents: Tensor,
    starts: Tensor,
    counts: Tensor,
    n_max: int,
    k: int,
) -> Tensor:
    """Assign batch-segmented flat points to their nearest centroid.

    Args:
        p_flat: Float32 `[N, D]` points, grouped by event.
        cents: Float32 `[B * k, D]` centroids.
        starts: Int64 `[B]` offset of each event in `p_flat`.
        counts: Int64 `[B]` number of points per event.
        n_max: Largest value in `counts`.
        k: Number of centroids per event.

    Returns:
        Int64 `[N]` indices in `[0, k)`; ties pick the lowest index.
    """
    p_flat, cents = p_flat.detach(), cents.detach()
    if p_flat.numel() == 0 or starts.numel() == 0:
        return torch.empty(p_flat.shape[0], device=p_flat.device).long()
    seg = torch.repeat_interleave(
        torch.arange(starts.numel(), device=p_flat.device),
        counts,
        output_size=p_flat.shape[0],
    )
    cents = cents.float().view(starts.numel(), k, -1).index_select(0, seg)
    return _sq_dist(p_flat.float().unsqueeze(1), cents).squeeze(1).argmin(-1)
