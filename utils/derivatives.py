from __future__ import annotations

import numpy as np

Array = np.ndarray


def _resample_closed_curve_uniform_arc_length(points: Array, num_points: int) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if len(points) < 4:
        raise ValueError("Need at least 4 points to resample a closed curve.")
    if num_points < 8:
        raise ValueError("num_points must be at least 8.")

    extended = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate curve with near-zero length.")

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, num_points, endpoint=False)

    out = np.empty((num_points, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] <= s:
            j += 1
        if j >= len(seg):
            j = len(seg) - 1
        local_len = seg[j]
        if local_len <= 1e-12:
            out[i] = extended[j]
        else:
            alpha = (s - cum[j]) / local_len
            out[i] = (1.0 - alpha) * extended[j] + alpha * extended[j + 1]
    return out


def _nearest_index(points: Array, query: Array) -> int:
    d2 = ((points - query.reshape(1, 2)) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def compute_euclidean_arc_length_derivatives(
    curve_points: Array,
    anchor_index: int,
    *,
    dense_num_points: int = 4096,
) -> tuple[Array, Array, Array]:
    """
    Compute Euclidean first and second derivatives at a sparse anchor index
    using a very dense, deterministic arc-length resampling of the full closed curve.

    Returns:
        first_ds:  (2,) first derivative d gamma / ds (unit tangent approximately)
        second_ds: (2,) second derivative d^2 gamma / ds^2 (curvature vector)
        anchor_dense_point: (2,) dense point used for the estimate
    """
    curve_points = np.asarray(curve_points, dtype=np.float64)
    if curve_points.ndim != 2 or curve_points.shape[1] != 2:
        raise ValueError("curve_points must have shape (N, 2).")
    if not (0 <= anchor_index < len(curve_points)):
        raise ValueError("anchor_index out of range.")

    dense = _resample_closed_curve_uniform_arc_length(curve_points, num_points=dense_num_points)
    q = curve_points[anchor_index]
    k = _nearest_index(dense, q)

    ds = 1.0 / dense_num_points
    # actual scale is total_length * ds, but first derivative direction is invariant.
    # For consistency of first/second supervision, use physical ds:
    extended = np.vstack([curve_points, curve_points[:1]])
    total_length = float(np.linalg.norm(np.diff(extended, axis=0), axis=1).sum())
    ds = total_length / dense_num_points

    prev_pt = dense[(k - 1) % dense_num_points]
    curr_pt = dense[k]
    next_pt = dense[(k + 1) % dense_num_points]

    first = (next_pt - prev_pt) / (2.0 * ds)
    first_norm = np.linalg.norm(first)
    if first_norm > 1e-12:
        first = first / first_norm

    second = (next_pt - 2.0 * curr_pt + prev_pt) / (ds ** 2)
    return first.astype(np.float64), second.astype(np.float64), curr_pt.astype(np.float64)
