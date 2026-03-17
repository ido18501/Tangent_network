from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from . import CanonicalCurve, RawCurve


def cumulative_arclength(points: np.ndarray, closed: bool = False) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths, dtype=np.float32)])
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        closing = np.linalg.norm(pts[0] - pts[-1])
        s = np.concatenate([s, [s[-1] + closing]]).astype(np.float32)
    return s.astype(np.float32)


def uniform_resample(points: np.ndarray, num_points: int, closed: bool = False) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return pts.copy()

    work = pts
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        work = np.vstack([pts, pts[:1]])

    s = cumulative_arclength(work, closed=False)
    total = float(s[-1])
    if total <= 1e-8:
        return np.repeat(work[:1], num_points, axis=0).astype(np.float32)

    if closed:
        target = np.linspace(0.0, total, num_points, endpoint=False, dtype=np.float32)
    else:
        target = np.linspace(0.0, total, num_points, endpoint=True, dtype=np.float32)

    x = np.interp(target, s, work[:, 0])
    y = np.interp(target, s, work[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def smooth_curve(points: np.ndarray, closed: bool = False, window: int = 11, polyorder: int = 3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < max(window, polyorder + 2):
        return pts.copy()

    if window % 2 == 0:
        window += 1
    window = min(window, len(pts) - (1 - len(pts) % 2))
    if window < polyorder + 2:
        return pts.copy()

    if closed:
        pad = window // 2
        ext = np.vstack([pts[-pad:], pts, pts[:pad]])
        xs = savgol_filter(ext[:, 0], window_length=window, polyorder=polyorder, mode="interp")
        ys = savgol_filter(ext[:, 1], window_length=window, polyorder=polyorder, mode="interp")
        sm = np.stack([xs, ys], axis=1)[pad:-pad]
    else:
        xs = savgol_filter(pts[:, 0], window_length=window, polyorder=polyorder, mode="interp")
        ys = savgol_filter(pts[:, 1], window_length=window, polyorder=polyorder, mode="interp")
        sm = np.stack([xs, ys], axis=1)
    return sm.astype(np.float32)


def canonicalize_curve(
    curve: RawCurve,
    *,
    dense_points: int = 512,
    smoothing: bool = True,
    target_extent: float = 2.0,
) -> CanonicalCurve:
    pts = np.asarray(curve.points, dtype=np.float32)
    resampled = uniform_resample(pts, num_points=dense_points, closed=curve.closed)
    if smoothing:
        resampled = smooth_curve(resampled, closed=curve.closed, window=11, polyorder=3)
        resampled = uniform_resample(resampled, num_points=dense_points, closed=curve.closed)

    center = resampled.mean(axis=0)
    centered = resampled - center
    extent = centered.max(axis=0) - centered.min(axis=0)
    max_extent = float(np.max(extent))
    scale = float(target_extent / max(max_extent, 1e-6))
    canonical = centered * scale

    normalization = {
        "center": center.astype(np.float32),
        "scale": np.float32(scale),
        "target_extent": np.float32(target_extent),
        "dense_points": int(dense_points),
        "closed": bool(curve.closed),
    }
    return CanonicalCurve(
        image_points=resampled.astype(np.float32),
        canonical_points=canonical.astype(np.float32),
        closed=curve.closed,
        normalization=normalization,
    )
