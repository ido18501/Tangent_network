from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.curve_generation import generate_random_simple_fourier_curve
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
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
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

    def _generate_curve(self, rng: np.random.Generator) -> Array:
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
        return curve_points

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