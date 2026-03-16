from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    generate_random_piecewise_curve,
    fit_curve_to_canvas_with_random_size,
    warp_curve_sampling,
)
from utils.real_contours import (
    RealContourLibrary,
    preprocess_real_contour_for_training,
)
from datasets.tangent_tuple_generation import build_random_tangent_training_tuple


Array = np.ndarray


@dataclass
class TangentSampleTensors:
    """
    One training sample returned as tensors.

    Shapes:
        anchor:    (patch_size, 2)
        positive:  (patch_size, 2)
        negatives: (num_negatives, patch_size, 2)
    """
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor
    transform_matrix: torch.Tensor
    family: str
    anchor_center_index: int
    negative_center_indices: torch.Tensor


class TangentDataset(Dataset):
    """
    On-the-fly dataset for tangent-learning tuples.

    v2:
        - optional variable half-width
        - optional cross-curve negatives
        - optional mild coordinate noise
    """

    def __init__(
        self,
        *,
        length: int,
        family: str,
        num_curve_points: int = 300,
        fourier_max_freq: int = 5,
        fourier_scale: float = 0.9,
        fourier_decay_power: float = 2.0,
        curve_max_tries: int = 300,
        curve_min_size: float = 0.45,
        curve_max_size: float = 0.75,
        patch_size: int = 9,
        half_width: int = 12,
        half_width_range: tuple[int, int] | None = None,
        num_negatives: int = 8,
        negative_min_offset: int = 5,
        negative_max_offset: int = 25,
        negative_other_curve_fraction: float = 0.5,
        patch_mode: str = "jittered_symmetric",
        jitter_fraction: float = 0.25,
        closed: bool = True,
        transform_kwargs: dict[str, Any] | None = None,
        return_centered: bool = True,
        point_noise_std: float = 0.0,
        curve_family_probs: dict[str, float] | None = None,
        warp_sampling_prob: float = 0.7,
        warp_sampling_strength: float = 0.18,
        orthogonal_noise_std: float = 0.0,
        real_curve_fraction: float = 0.0,
        real_contours_npz_dir: str | None = None,
        real_closed_only: bool = True,
        real_closed_threshold: float = 1.5,
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
        if not (0.0 <= real_curve_fraction <= 1.0):
            raise ValueError("real_curve_fraction must be in [0, 1].")

        if real_curve_fraction > 0.0 and real_contours_npz_dir is None:
            raise ValueError(
                "real_contours_npz_dir must be provided when real_curve_fraction > 0."
            )
        if not (0.0 <= warp_sampling_prob <= 1.0):
            raise ValueError("warp_sampling_prob must be in [0, 1].")
        if warp_sampling_strength < 0.0:
            raise ValueError("warp_sampling_strength must be nonnegative.")
        if orthogonal_noise_std < 0.0:
            raise ValueError("orthogonal_noise_std must be nonnegative.")
        if curve_family_probs is None:
            curve_family_probs = {"fourier": 0.55, "piecewise": 0.45}

        self.curve_family_probs = dict(curve_family_probs)
        self.warp_sampling_prob = warp_sampling_prob
        self.warp_sampling_strength = warp_sampling_strength
        self.orthogonal_noise_std = orthogonal_noise_std
        if length < 1:
            raise ValueError("length must be at least 1.")
        if num_curve_points < 20:
            raise ValueError("num_curve_points should be reasonably large.")
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be odd and at least 3.")
        if half_width < 1:
            raise ValueError("half_width must be at least 1.")
        if num_negatives < 1:
            raise ValueError("num_negatives must be at least 1.")
        if negative_min_offset < 1 or negative_max_offset < negative_min_offset:
            raise ValueError("Require 1 <= negative_min_offset <= negative_max_offset.")
        if not (0.0 <= negative_other_curve_fraction <= 1.0):
            raise ValueError("negative_other_curve_fraction must be in [0, 1].")
        if point_noise_std < 0.0:
            raise ValueError("point_noise_std must be nonnegative.")

        self.length = length
        self.family = family


        self.real_curve_fraction = real_curve_fraction
        self.real_contours_npz_dir = real_contours_npz_dir
        self.real_closed_only = real_closed_only
        self.real_closed_threshold = real_closed_threshold
        self.real_contour_library: RealContourLibrary | None = None
        if self.real_curve_fraction > 0.0:
            self.real_contour_library = RealContourLibrary(
                contour_dir=self.real_contours_npz_dir,
                min_points=30,
                closed_threshold=self.real_closed_threshold,
                closed_only=self.real_closed_only,
            )
        self.num_curve_points = num_curve_points
        self.fourier_max_freq = fourier_max_freq
        self.fourier_scale = fourier_scale
        self.fourier_decay_power = fourier_decay_power
        self.curve_max_tries = curve_max_tries
        self.curve_min_size = curve_min_size
        self.curve_max_size = curve_max_size

        self.patch_size = patch_size
        self.half_width = half_width
        self.half_width_range = half_width_range

        self.num_negatives = num_negatives
        self.negative_min_offset = negative_min_offset
        self.negative_max_offset = negative_max_offset
        self.negative_other_curve_fraction = negative_other_curve_fraction

        self.patch_mode = patch_mode
        self.jitter_fraction = jitter_fraction
        self.closed = closed

        self.transform_kwargs = {} if transform_kwargs is None else dict(transform_kwargs)
        self.return_centered = return_centered
        self.point_noise_std = point_noise_std
        self.dtype = dtype

        self._base_seed = seed

    def __len__(self) -> int:
        return self.length

    def _make_rng(self, index: int) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._base_seed + index)
    def _sample_curve_family(self, rng: np.random.Generator) -> str:
        names = list(self.curve_family_probs.keys())
        probs = np.asarray([self.curve_family_probs[n] for n in names], dtype=np.float64)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def _add_curve_noise(self, curve_points: Array, rng: np.random.Generator) -> Array:
        pts = np.asarray(curve_points, dtype=np.float64).copy()

        if self.point_noise_std > 0.0:
            pts += rng.normal(0.0, self.point_noise_std, size=pts.shape)

        if self.orthogonal_noise_std > 0.0:
            prev_pts = np.roll(pts, 1, axis=0)
            next_pts = np.roll(pts, -1, axis=0)
            tang = next_pts - prev_pts
            tang_norm = np.linalg.norm(tang, axis=1, keepdims=True)
            tang = tang / np.clip(tang_norm, 1e-12, None)
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
            coeff = rng.normal(0.0, self.orthogonal_noise_std, size=(len(pts), 1))
            pts = pts + coeff * normal

        return pts

    def _generate_synthetic_curve(self, rng: np.random.Generator) -> Array:
        family = self._sample_curve_family(rng)

        if family == "fourier":
            t = np.linspace(0.0, 2.0 * np.pi, self.num_curve_points, endpoint=False)

            curve_points, _ = generate_random_simple_fourier_curve(
                t=t,
                max_freq=self.fourier_max_freq,
                scale=self.fourier_scale,
                decay_power=self.fourier_decay_power,
                rng=rng,
                max_tries=self.curve_max_tries,
                center=True,
                fit_to_canvas=True,
                min_size=self.curve_min_size,
                max_size=self.curve_max_size,
            )

        elif family == "piecewise":
            curve_points = generate_random_piecewise_curve(
                num_points=self.num_curve_points,
                rng=rng,
                closed=self.closed,
            )
            curve_points = fit_curve_to_canvas_with_random_size(
                curve_points,
                rng=rng,
                min_size=self.curve_min_size,
                max_size=self.curve_max_size,
            )

        else:
            raise ValueError(f"Unsupported sampled curve family: {family}")

        if rng.random() < self.warp_sampling_prob:
            curve_points = warp_curve_sampling(
                curve_points,
                rng=rng,
                strength=self.warp_sampling_strength,
                closed=self.closed,
            )

        curve_points = self._add_curve_noise(curve_points, rng)
        return curve_points

    def _generate_real_curve(self, rng: np.random.Generator) -> Array:
        if self.real_contour_library is None:
            raise RuntimeError("Real contour library is not initialized.")

        raw_contour = self.real_contour_library.sample_raw_contour(rng)

        curve_points = preprocess_real_contour_for_training(
            raw_contour,
            num_curve_points=self.num_curve_points,
            rng=rng,
            closed=self.closed,
            curve_min_size=self.curve_min_size,
            curve_max_size=self.curve_max_size,
        )

        if rng.random() < self.warp_sampling_prob:
            curve_points = warp_curve_sampling(
                curve_points,
                rng=rng,
                strength=self.warp_sampling_strength,
                closed=self.closed,
            )

        curve_points = self._add_curve_noise(curve_points, rng)
        return curve_points

    def _generate_curve(self, rng: np.random.Generator) -> Array:
        use_real = (
                self.real_contour_library is not None
                and rng.random() < self.real_curve_fraction
        )

        if use_real:
            return self._generate_real_curve(rng)

        return self._generate_synthetic_curve(rng)

    def _sample_half_width(self, rng: np.random.Generator) -> int:
        if self.half_width_range is None:
            return self.half_width

        lo, hi = self.half_width_range
        if lo < 1 or hi < lo:
            raise ValueError("Invalid half_width_range.")

        return int(rng.integers(lo, hi + 1))

    def _maybe_add_noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.point_noise_std <= 0.0:
            return x
        noise = rng.normal(loc=0.0, scale=self.point_noise_std, size=x.shape)
        return x + noise.astype(np.float64)

    def __getitem__(self, index: int) -> TangentSampleTensors:
        rng = self._make_rng(index)

        curve_points = self._generate_curve(rng)
        current_half_width = self._sample_half_width(rng)

        num_cross_curve_negatives = int(round(self.num_negatives * self.negative_other_curve_fraction))
        num_cross_curve_negatives = min(num_cross_curve_negatives, self.num_negatives)

        external_negative_curves = [
            self._generate_curve(rng) for _ in range(num_cross_curve_negatives)
        ]

        tuple_sample = build_random_tangent_training_tuple(
            curve_points=curve_points,
            family=self.family,
            patch_size=self.patch_size,
            half_width=current_half_width,
            num_negatives=self.num_negatives,
            negative_min_offset=self.negative_min_offset,
            negative_max_offset=self.negative_max_offset,
            closed=self.closed,
            patch_mode=self.patch_mode,
            jitter_fraction=self.jitter_fraction,
            rng=rng,
            transform_kwargs=self.transform_kwargs,
            external_negative_curves=external_negative_curves,
            num_cross_curve_negatives=num_cross_curve_negatives,
        )

        if self.return_centered:
            anchor_np = tuple_sample.anchor.centered_points
            positive_np = tuple_sample.positive.centered_points
            negatives_np = np.stack(
                [neg.centered_points for neg in tuple_sample.negatives],
                axis=0,
            )
        else:
            anchor_np = tuple_sample.anchor.points
            positive_np = tuple_sample.positive.points
            negatives_np = np.stack(
                [neg.points for neg in tuple_sample.negatives],
                axis=0,
            )

        anchor_np = self._maybe_add_noise(anchor_np, rng)
        positive_np = self._maybe_add_noise(positive_np, rng)
        negatives_np = self._maybe_add_noise(negatives_np, rng)

        anchor = torch.as_tensor(anchor_np, dtype=self.dtype)
        positive = torch.as_tensor(positive_np, dtype=self.dtype)
        negatives = torch.as_tensor(negatives_np, dtype=self.dtype)
        negative_center_indices = torch.as_tensor(
            tuple_sample.negative_center_indices,
            dtype=torch.long,
        )
        transform_matrix = torch.as_tensor(
            tuple_sample.transform.A,
            dtype=self.dtype,
        )

        return TangentSampleTensors(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            transform_matrix=transform_matrix,
            family=tuple_sample.family,
            anchor_center_index=tuple_sample.anchor_center_index,
            negative_center_indices=negative_center_indices,
        )