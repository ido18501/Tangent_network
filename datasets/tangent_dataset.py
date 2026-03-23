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
from datasets.tangent_tuple_generation import build_random_tangent_training_tuple

Array = np.ndarray


@dataclass
class TangentSampleTensors:
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor
    transform_matrix: torch.Tensor
    family: str
    anchor_center_index: int
    negative_center_indices: torch.Tensor
    gt_first_anchor: torch.Tensor
    gt_second_anchor: torch.Tensor


class TangentDataset(Dataset):
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
        patch_mode: str = "random_warp_symmetric",
        jitter_fraction: float = 0.25,
        closed: bool = True,
        transform_kwargs: dict[str, Any] | None = None,
        return_centered: bool = True,
        point_noise_std: float = 0.0,
        curve_family_probs: dict[str, float] | None = None,
        warp_sampling_prob: float = 0.7,
        warp_sampling_strength: float = 0.18,
        orthogonal_noise_std: float = 0.0,
        gt_dense_num_points: int = 4096,
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
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
        self.curve_family_probs = dict(curve_family_probs or {"fourier": 1.0, "piecewise": 0.0})
        self.warp_sampling_prob = warp_sampling_prob
        self.warp_sampling_strength = warp_sampling_strength
        self.orthogonal_noise_std = orthogonal_noise_std
        self.gt_dense_num_points = gt_dense_num_points
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

    def _generate_curve(self, rng: np.random.Generator) -> Array:
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

    def _sample_half_width(self, rng: np.random.Generator) -> int:
        if self.half_width_range is None:
            return self.half_width
        low, high = self.half_width_range
        return int(rng.integers(low, high + 1))

    def __getitem__(self, index: int) -> TangentSampleTensors:
        rng = self._make_rng(index)
        curve_points = self._generate_curve(rng)

        num_cross_curve_negatives = int(round(self.num_negatives * self.negative_other_curve_fraction))
        external_negative_curves: list[Array] = []
        for _ in range(num_cross_curve_negatives):
            external_negative_curves.append(self._generate_curve(rng))

        half_width = self._sample_half_width(rng)

        tuple_sample = build_random_tangent_training_tuple(
            curve_points=curve_points,
            transform_family=self.family,
            patch_size=self.patch_size,
            half_width=half_width,
            num_negatives=self.num_negatives,
            negative_min_offset=self.negative_min_offset,
            negative_max_offset=self.negative_max_offset,
            closed=self.closed,
            patch_mode=self.patch_mode,
            jitter_fraction=self.jitter_fraction,
            rng=rng,
            transform_kwargs=self.transform_kwargs,
            external_negative_curves=external_negative_curves if num_cross_curve_negatives > 0 else None,
            num_cross_curve_negatives=num_cross_curve_negatives,
            gt_dense_num_points=self.gt_dense_num_points,
        )

        anchor = torch.as_tensor(tuple_sample.anchor_patch, dtype=self.dtype)
        positive = torch.as_tensor(tuple_sample.positive_patch, dtype=self.dtype)
        negatives = torch.as_tensor(tuple_sample.negative_patches, dtype=self.dtype)
        transform_matrix = torch.as_tensor(tuple_sample.transform_matrix, dtype=self.dtype)
        gt_first_anchor = torch.as_tensor(tuple_sample.gt_first_anchor, dtype=self.dtype)
        gt_second_anchor = torch.as_tensor(tuple_sample.gt_second_anchor, dtype=self.dtype)

        if self.return_centered:
            anchor_center = anchor[self.patch_size // 2].clone()
            positive_center = positive[self.patch_size // 2].clone()
            anchor = anchor - anchor_center.unsqueeze(0)
            positive = positive - positive_center.unsqueeze(0)
            negatives = negatives - negatives[:, self.patch_size // 2, :].unsqueeze(1)

        return TangentSampleTensors(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            transform_matrix=transform_matrix,
            family=tuple_sample.family,
            anchor_center_index=tuple_sample.anchor_center_index,
            negative_center_indices=torch.as_tensor(tuple_sample.negative_center_indices, dtype=torch.long),
            gt_first_anchor=gt_first_anchor,
            gt_second_anchor=gt_second_anchor,
        )
