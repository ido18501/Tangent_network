from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

from utils.curve_generation import (
    center_curve,
    fit_curve_to_canvas_with_random_size,
    resample_polyline_uniform,
)

Array = np.ndarray


def load_contours_npz(path: str | Path) -> list[Array]:
    data = np.load(path, allow_pickle=False)
    num_contours = int(data["num_contours"][0])
    return [np.asarray(data[f"contour_{i}"], dtype=np.float64) for i in range(num_contours)]


def remove_consecutive_duplicates(points: Array, eps: float = 1e-12) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points

    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > eps:
            keep.append(i)

    return points[np.asarray(keep, dtype=np.int64)]


def is_closed_like(points: Array, threshold: float = 1.5) -> bool:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return False
    return float(np.linalg.norm(points[0] - points[-1])) <= threshold


@dataclass
class RealContourLibrary:
    contour_dir: str
    min_points: int = 30
    closed_threshold: float = 1.5
    closed_only: bool = True

    def __post_init__(self) -> None:
        root = Path(self.contour_dir)
        if not root.exists():
            raise FileNotFoundError(f"Real contour directory not found: {root}")

        self.contours: list[Array] = []

        for npz_path in sorted(root.glob("*_contours.npz")):
            contours = load_contours_npz(npz_path)
            for c in contours:
                c = np.asarray(c, dtype=np.float64)

                if c.ndim != 2 or c.shape[1] != 2:
                    continue

                c = remove_consecutive_duplicates(c)

                if len(c) < self.min_points:
                    continue

                closed_like = is_closed_like(c, threshold=self.closed_threshold)
                if self.closed_only and not closed_like:
                    continue

                self.contours.append(c)

        if len(self.contours) == 0:
            raise RuntimeError(
                f"No usable real contours were loaded from {root}."
            )

    def __len__(self) -> int:
        return len(self.contours)

    def sample_raw_contour(self, rng: np.random.Generator) -> Array:
        idx = int(rng.integers(0, len(self.contours)))
        return np.asarray(self.contours[idx], dtype=np.float64).copy()


def preprocess_real_contour_for_training(
    contour_xy: Array,
    *,
    num_curve_points: int,
    rng: np.random.Generator,
    closed: bool = True,
    curve_min_size: float = 0.25,
    curve_max_size: float = 0.95,
) -> Array:
    contour_xy = np.asarray(contour_xy, dtype=np.float64)

    contour_xy = remove_consecutive_duplicates(contour_xy)

    if len(contour_xy) < 3:
        raise ValueError("Real contour is too short after duplicate removal.")

    contour_xy = resample_polyline_uniform(
        contour_xy,
        num_points=num_curve_points,
        closed=closed,
    )

    contour_xy = center_curve(contour_xy)

    contour_xy = fit_curve_to_canvas_with_random_size(
        contour_xy,
        rng=rng,
        min_size=curve_min_size,
        max_size=curve_max_size,
    )

    return contour_xy
