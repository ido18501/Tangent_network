from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters

from . import RawCurve


def curve_arc_length(points: np.ndarray, closed: bool = False) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        length += float(np.linalg.norm(pts[0] - pts[-1]))
    return length


def _curvature_proxy(points: np.ndarray, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return np.zeros(0, dtype=np.float64)

    if closed:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        center = pts
    else:
        prev_pts = pts[:-2]
        center = pts[1:-1]
        next_pts = pts[2:]

    v1 = center - prev_pts
    v2 = next_pts - center
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    denom = 0.5 * (n1 + n2)
    return ang / np.maximum(denom, 1e-6)


def _sharp_turn_fraction(points: np.ndarray, closed: bool, degrees: float = 75.0) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return 1.0
    threshold = math.radians(degrees)

    if closed:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        center = pts
    else:
        prev_pts = pts[:-2]
        center = pts[1:-1]
        next_pts = pts[2:]

    v1 = center - prev_pts
    v2 = next_pts - center
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.mean(ang > threshold)) if len(ang) else 1.0


def _turning_density(points: np.ndarray, closed: bool) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return 1e9

    if closed:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        center = pts
    else:
        prev_pts = pts[:-2]
        center = pts[1:-1]
        next_pts = pts[2:]

    v1 = center - prev_pts
    v2 = next_pts - center
    n1 = np.linalg.norm(v1, axis=1) + 1e-8
    n2 = np.linalg.norm(v2, axis=1) + 1e-8
    cosang = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)

    total_turn = float(np.sum(np.abs(ang)))
    length = curve_arc_length(points, closed)
    return total_turn / max(length, 1e-6)

def _touches_border(points: np.ndarray, image_shape: tuple[int, int], margin: int = 2) -> bool:
    h, w = image_shape[:2]
    x = points[:, 0]
    y = points[:, 1]
    return bool(
        np.any(x <= margin)
        or np.any(y <= margin)
        or np.any(x >= (w - 1 - margin))
        or np.any(y >= (h - 1 - margin))
    )

def _linearity_ratio(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 1.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-8))
    return float(eigvals[1] / eigvals[0])

def _segments_intersect(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    eps = 1e-8
    if abs(o1) < eps and on_segment(a1, b1, a2):
        return True
    if abs(o2) < eps and on_segment(a1, b2, a2):
        return True
    if abs(o3) < eps and on_segment(b1, a1, b2):
        return True
    if abs(o4) < eps and on_segment(b1, a2, b2):
        return True
    return False


def has_self_intersections(points: np.ndarray, closed: bool) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 8:
        return False
    segs = list(zip(pts[:-1], pts[1:]))
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        segs.append((pts[-1], pts[0]))

    for i, (a1, a2) in enumerate(segs):
        for j in range(i + 1, len(segs)):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and j == len(segs) - 1:
                continue
            b1, b2 = segs[j]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _contrast_support(points: np.ndarray, image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    grad = filters.scharr(gray)
    grad = ndi.gaussian_filter(grad, sigma=1.0)
    rr = np.clip(np.round(points[:, 1]).astype(int), 0, grad.shape[0] - 1)
    cc = np.clip(np.round(points[:, 0]).astype(int), 0, grad.shape[1] - 1)
    return float(np.mean(grad[rr, cc]))


def score_curve(curve: RawCurve, image: np.ndarray) -> float:
    pts = curve.points
    image_shape = image.shape[:2]
    h, w = image_shape
    image_diag = float(np.hypot(h, w))

    length = curve_arc_length(pts, curve.closed)
    if length <= 1e-6:
        return -1.0

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    bbox_diag = float(np.linalg.norm(max_xy - min_xy))

    bbox_w = float(max_xy[0] - min_xy[0])
    bbox_h = float(max_xy[1] - min_xy[1])
    bbox_area = max(bbox_w * bbox_h, 1e-6)
    length_density = length / (bbox_area ** 0.5 + 1e-6)
    density_score = np.clip(length_density / 6.0, 0.0, 1.2)

    extent_score = np.clip(bbox_diag / (0.25 * image_diag), 0.0, 1.6)
    length_score = np.clip(length / (0.55 * image_diag), 0.0, 2.0)

    curv = _curvature_proxy(pts, curve.closed)
    curvature_var = float(np.var(curv)) if len(curv) else 1e9
    sharp_frac = _sharp_turn_fraction(pts, curve.closed)
    turn_density = _turning_density(pts, curve.closed)

    smooth_score = float(np.exp(-4.0 * min(curvature_var, 1.5)))
    turn_score = float(np.exp(-5.0 * min(turn_density, 0.5)))
    sharp_score = max(0.0, 1.0 - 4.0 * sharp_frac)

    border_penalty = 1.0 if _touches_border(pts, image_shape, margin=6) else 0.0
    self_intersection_penalty = 1.0 if has_self_intersections(pts, curve.closed) else 0.0

    contrast = _contrast_support(pts, image)
    contrast_score = np.clip(contrast / 0.10, 0.0, 1.2)

    source_bonus = 0.25 if curve.source == "region_boundary" else 0.0
    closed_bonus = 0.55 if curve.closed else 0.0
    linearity = _linearity_ratio(pts)
    linearity_penalty = np.clip((linearity - 18.0) / 20.0, 0.0, 1.0)
    interior_std = float(curve.metadata.get("interior_std", 0.0))
    texture_penalty = np.clip((interior_std - 0.10) / 0.20, 0.0, 1.0)
    region_area = float(curve.metadata.get("region_area", 0.0))
    image_area = float(image.shape[0] * image.shape[1])
    area_score = np.clip(region_area / (0.03 * image_area), 0.0, 1.2)
    score = (
        2.8 * length_score
        + 2.0 * extent_score
        + 2.2 * smooth_score
        + 1.6 * turn_score
        + 1.2 * sharp_score
        + 0.8 * contrast_score
        + 0.8 * float(curve.confidence)
        + source_bonus
        - 1.5 * border_penalty
        - 2 * self_intersection_penalty
        + closed_bonus
        + 0.9 * density_score
        - 0.25 * texture_penalty
        - 0.6 * linearity_penalty
        + 1.1 * area_score
    )
    return float(score)


def filter_curves(
    curves: Iterable[RawCurve],
    image: np.ndarray,
    *,
    min_arc_length: float = 46.0,
    min_extent: float = 14.0,
    top_k: int = 32,
) -> list[tuple[RawCurve, float]]:
    filtered: list[tuple[RawCurve, float]] = []
    for curve in curves:
        pts = curve.points
        if len(pts) < 16:
            continue
        length = curve_arc_length(pts, curve.closed)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        extent = float(np.linalg.norm(max_xy - min_xy))
        curv = _curvature_proxy(pts, curve.closed)
        curvature_var = float(np.var(curv)) if len(curv) else 1e9
        sharp_frac = _sharp_turn_fraction(pts, curve.closed)
        turn_density = _turning_density(pts, curve.closed)

        if curvature_var > 2.2:
            continue
        if sharp_frac > 0.58:
            continue
        if turn_density > 0.36:
            continue
        if has_self_intersections(pts, curve.closed):
            continue
        effective_min_arc = min_arc_length if curve.closed else 56.0
        if length < effective_min_arc:
            continue
        if _touches_border(pts, image.shape[:2], margin=0):
            continue
        score = score_curve(curve, image)
        if score <= 0.0:
            continue
        filtered.append((curve, score))

    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[:top_k]
