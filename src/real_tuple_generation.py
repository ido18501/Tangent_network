from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.patch_sampling import CurvePatchSample
from utils.transformations import Transformation2D, apply_transformation, sample_transformation
from src.real_patch_adapter import (
    CanonicalContour,
    canonicalize_real_contour,
    get_valid_center_indices,
    sample_real_patch_at_center,
)

Array = np.ndarray


@dataclass
class RealTangentTrainingTuple:
    family: str
    anchor: CurvePatchSample
    positive: CurvePatchSample
    negatives: list[CurvePatchSample]
    transform: Transformation2D
    anchor_center_index: int
    negative_center_indices: Array
    closed: bool


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def _sample_local_negative_indices(
    num_points: int,
    anchor_center_index: int,
    num_negatives: int,
    min_offset: int,
    max_offset: int,
    closed: bool,
    rng: np.random.Generator,
) -> Array:
    possible_offsets = np.concatenate([
        -np.arange(min_offset, max_offset + 1, dtype=np.int64),
         np.arange(min_offset, max_offset + 1, dtype=np.int64),
    ])

    sampled_offsets = rng.choice(possible_offsets, size=num_negatives, replace=True)
    candidate_indices = anchor_center_index + sampled_offsets

    if closed:
        candidate_indices = np.mod(candidate_indices, num_points)
    else:
        candidate_indices = np.clip(candidate_indices, 0, num_points - 1)
        for i in range(len(candidate_indices)):
            if candidate_indices[i] == anchor_center_index:
                if anchor_center_index + min_offset < num_points:
                    candidate_indices[i] = anchor_center_index + min_offset
                elif anchor_center_index - min_offset >= 0:
                    candidate_indices[i] = anchor_center_index - min_offset
                else:
                    raise ValueError("Could not find valid open-curve negative index.")

    return candidate_indices.astype(np.int64)


def build_real_tangent_training_tuple(
    contour_xy: Array,
    *,
    family: str,
    dense_num_points: int,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    patch_mode: str = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    anchor_center_index: int | None = None,
    closed: bool | None = None,
    target_extent: float = 0.6,
    rng: np.random.Generator | None = None,
    transform_kwargs: dict[str, Any] | None = None,
) -> RealTangentTrainingTuple:
    rng = _ensure_rng(rng)
    if transform_kwargs is None:
        transform_kwargs = {}

    base_curve = canonicalize_real_contour(
        contour_xy,
        dense_num_points=dense_num_points,
        closed=closed,
        target_extent=target_extent,
    )

    valid_centers = get_valid_center_indices(
        num_points=len(base_curve.canonical_points),
        half_width=half_width,
        closed=base_curve.closed,
    )
    if len(valid_centers) == 0:
        raise ValueError("No valid center indices for this contour.")

    if anchor_center_index is None:
        anchor_center_index = int(rng.choice(valid_centers))
    else:
        anchor_center_index = int(anchor_center_index)

    # Anchor on canonical real contour
    anchor = sample_real_patch_at_center(
        base_curve,
        anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
    )

    # Positive from transformed dense contour, independently re-sampled
    transform = sample_transformation(
        family=family,
        rng=rng,
        **transform_kwargs,
    )

    transformed_dense = apply_transformation(base_curve.canonical_points, transform)

    transformed_curve = CanonicalContour(
        image_points=base_curve.image_points,   # unused for patch construction
        canonical_points=transformed_dense,
        closed=base_curve.closed,
        normalization=base_curve.normalization,
    )

    positive = sample_real_patch_at_center(
        transformed_curve,
        anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
    )

    negative_center_indices = _sample_local_negative_indices(
        num_points=len(base_curve.canonical_points),
        anchor_center_index=anchor_center_index,
        num_negatives=num_negatives,
        min_offset=negative_min_offset,
        max_offset=negative_max_offset,
        closed=base_curve.closed,
        rng=rng,
    )

    negatives = []
    for neg_idx in negative_center_indices:
        neg_patch = sample_real_patch_at_center(
            transformed_curve,
            int(neg_idx),
            patch_size=patch_size,
            half_width=half_width,
            patch_mode=patch_mode,
            jitter_fraction=jitter_fraction,
            rng=rng,
        )
        negatives.append(neg_patch)

    return RealTangentTrainingTuple(
        family=family,
        anchor=anchor,
        positive=positive,
        negatives=negatives,
        transform=transform,
        anchor_center_index=anchor_center_index,
        negative_center_indices=negative_center_indices,
        closed=base_curve.closed,
    )
