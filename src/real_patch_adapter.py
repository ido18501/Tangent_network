from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from utils.patch_sampling import CurvePatchSample, sample_patch_around_index
from utils.curve_generation import get_max_abs_extent

Array = np.ndarray
PatchSamplingMode = Literal["uniform_symmetric", "jittered_symmetric"]


@dataclass
class ContourNormalization:
    center: Array          # shape (2,)
    scale: float           # canonical = (x - center) * scale
    target_extent: float


@dataclass
class CanonicalContour:
    image_points: Array        # dense contour in image coords, shape (M, 2)
    canonical_points: Array    # dense contour in training-like coords, shape (M, 2)
    closed: bool
    normalization: ContourNormalization


def is_closed_contour(contour_xy: Array, closure_tol: float = 5.0) -> bool:
    contour_xy = np.asarray(contour_xy, dtype=np.float64)
    if len(contour_xy) < 3:
        return False
    return np.linalg.norm(contour_xy[0] - contour_xy[-1]) <= closure_tol


def compute_polyline_arclength(points: Array, closed: bool) -> tuple[Array, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {points.shape}")
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")

    if closed:
        segs = np.roll(points, -1, axis=0) - points
        seg_lens = np.linalg.norm(segs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(seg_lens.sum())
        return s, total
    else:
        segs = points[1:] - points[:-1]
        seg_lens = np.linalg.norm(segs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(seg_lens.sum())
        return s, total


def resample_contour_uniform(points: Array, num_points: int, closed: bool) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {points.shape}")
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")

    s, total = compute_polyline_arclength(points, closed=closed)
    if total <= 1e-12:
        raise ValueError("Contour has near-zero length.")

    if closed:
        pts_ext = np.vstack([points, points[:1]])
        target_s = np.linspace(0.0, total, num_points, endpoint=False)
    else:
        pts_ext = points
        target_s = np.linspace(0.0, total, num_points, endpoint=True)

    x = np.interp(target_s, s, pts_ext[:, 0])
    y = np.interp(target_s, s, pts_ext[:, 1])
    return np.stack([x, y], axis=1)


def normalize_contour_to_training_canvas(
    points: Array,
    target_extent: float = 0.6,
) -> tuple[Array, ContourNormalization]:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center.reshape(1, 2)

    extent = get_max_abs_extent(centered)
    if extent <= 1e-12:
        raise ValueError("Contour extent too small to normalize.")

    scale = target_extent / extent
    canonical = centered * scale

    return canonical, ContourNormalization(
        center=center,
        scale=float(scale),
        target_extent=float(target_extent),
    )


def canonicalize_real_contour(
    contour_xy: Array,
    *,
    dense_num_points: int,
    closed: bool | None = None,
    target_extent: float = 0.6,
) -> CanonicalContour:
    contour_xy = np.asarray(contour_xy, dtype=np.float64)

    if closed is None:
        closed = is_closed_contour(contour_xy)

    dense_image = resample_contour_uniform(
        points=contour_xy,
        num_points=dense_num_points,
        closed=closed,
    )

    dense_canonical, norm = normalize_contour_to_training_canvas(
        dense_image,
        target_extent=target_extent,
    )

    return CanonicalContour(
        image_points=dense_image,
        canonical_points=dense_canonical,
        closed=closed,
        normalization=norm,
    )


def get_valid_center_indices(num_points: int, half_width: int, closed: bool) -> Array:
    if closed:
        return np.arange(num_points, dtype=np.int64)

    left = half_width
    right = num_points - half_width
    if left >= right:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(left, right, dtype=np.int64)


def sample_real_patch_at_center(
    canonical_curve: CanonicalContour,
    center_index: int,
    *,
    patch_size: int,
    half_width: int,
    patch_mode: PatchSamplingMode = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
) -> CurvePatchSample:
    if rng is None:
        rng = np.random.default_rng()

    return sample_patch_around_index(
        curve_points=canonical_curve.canonical_points,
        center_index=center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=canonical_curve.closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )
