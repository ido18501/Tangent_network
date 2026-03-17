from __future__ import annotations

import numpy as np


def _sample_positions(
    center_index: float,
    patch_size: int,
    *,
    spacing: float = 1.0,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    half = (patch_size - 1) / 2.0
    offsets = np.arange(patch_size, dtype=np.float32) - half
    positions = center_index + offsets * spacing
    if jitter > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        positions = positions + rng.uniform(-jitter, jitter, size=patch_size).astype(np.float32)
        positions.sort()
    return positions.astype(np.float32)


def _interpolate_curve(points: np.ndarray, positions: np.ndarray, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n == 0:
        raise ValueError("Curve must contain at least one point.")
    if closed:
        pos = np.mod(positions, n)
    else:
        pos = np.clip(positions, 0.0, n - 1.0)

    i0 = np.floor(pos).astype(int)
    i1 = (i0 + 1) % n if closed else np.clip(i0 + 1, 0, n - 1)
    alpha = (pos - i0).reshape(-1, 1).astype(np.float32)
    return ((1.0 - alpha) * pts[i0] + alpha * pts[i1]).astype(np.float32)


def sample_patch(
    curve: np.ndarray,
    center_index: int | float,
    patch_size: int,
    *,
    closed: bool = False,
    spacing: float = 1.0,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a local ordered patch with symmetric support around a center."""
    positions = _sample_positions(
        float(center_index),
        patch_size,
        spacing=spacing,
        jitter=jitter,
        rng=rng,
    )
    return _interpolate_curve(np.asarray(curve, dtype=np.float32), positions, closed)


def sample_positive_pair(
    curve: np.ndarray,
    center_index: int,
    patch_size: int,
    *,
    closed: bool = False,
    center_jitter: float = 1.5,
    spacing_jitter: float = 0.08,
    point_jitter: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng
    center_a = center_index + rng.uniform(-center_jitter, center_jitter)
    center_b = center_index + rng.uniform(-center_jitter, center_jitter)
    spacing_a = 1.0 + rng.uniform(-spacing_jitter, spacing_jitter)
    spacing_b = 1.0 + rng.uniform(-spacing_jitter, spacing_jitter)
    patch_a = sample_patch(curve, center_a, patch_size, closed=closed, spacing=spacing_a, jitter=point_jitter, rng=rng)
    patch_b = sample_patch(curve, center_b, patch_size, closed=closed, spacing=spacing_b, jitter=point_jitter, rng=rng)
    return patch_a, patch_b


def sample_negative_center(
    curve_length: int,
    center_index: int,
    min_separation: int,
    *,
    closed: bool = False,
    rng: np.random.Generator | None = None,
) -> int:
    rng = np.random.default_rng() if rng is None else rng
    if curve_length <= 1:
        return 0

    if closed:
        choices = [
            i for i in range(curve_length)
            if min((i - center_index) % curve_length, (center_index - i) % curve_length) >= min_separation
        ]
    else:
        choices = [i for i in range(curve_length) if abs(i - center_index) >= min_separation]

    if not choices:
        return int((center_index + curve_length // 2) % curve_length) if closed else int(np.clip(center_index + min_separation, 0, curve_length - 1))
    return int(rng.choice(np.array(choices, dtype=int)))
