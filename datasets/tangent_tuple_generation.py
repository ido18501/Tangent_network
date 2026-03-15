from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.patch_sampling import CurvePatchSample, sample_patch_around_index
from utils.transformations import Transformation2D, apply_transformation, sample_transformation


Array = np.ndarray


@dataclass
class TangentTrainingTuple:
    """
    A first-version training tuple for tangent learning.

    Attributes:
        family:
            Transformation family used to generate the positive / negatives.
        anchor:
            Patch sampled at the anchor center on the original curve.
        positive:
            Patch sampled at the same center on a transformed curve.
        negatives:
            List of patches sampled at nearby but different centers on the same
            transformed curve.
        transform:
            The sampled transformation used to create the transformed curve.
        anchor_center_index:
            Center index of the anchor point on the dense curve.
        negative_center_indices:
            Center indices used for the negative patches.
    """
    family: str
    anchor: CurvePatchSample
    positive: CurvePatchSample
    negatives: list[CurvePatchSample]
    transform: Transformation2D
    anchor_center_index: int
    negative_center_indices: Array


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
    """
    Sample nearby-but-different center indices for hard negatives.

    Negatives are chosen by integer offsets from the anchor center:
        +/- k, where k in [min_offset, max_offset]

    Args:
        num_points:
            Number of points in the dense curve.
        anchor_center_index:
            Anchor center index.
        num_negatives:
            Number of negatives to sample.
        min_offset:
            Minimum absolute offset from anchor.
        max_offset:
            Maximum absolute offset from anchor.
        closed:
            Whether the curve wraps around.
        rng:
            NumPy random generator.

    Returns:
        negative_center_indices:
            Array of shape (num_negatives,)
    """
    if num_negatives < 1:
        raise ValueError("num_negatives must be at least 1.")
    if min_offset < 1:
        raise ValueError("min_offset must be at least 1.")
    if max_offset < min_offset:
        raise ValueError("Require max_offset >= min_offset.")

    possible_offsets = np.concatenate([
        -np.arange(min_offset, max_offset + 1, dtype=np.int64),
         np.arange(min_offset, max_offset + 1, dtype=np.int64),
    ])

    if len(possible_offsets) == 0:
        raise ValueError("No possible negative offsets were generated.")

    sampled_offsets = rng.choice(
        possible_offsets,
        size=num_negatives,
        replace=True,
    )

    candidate_indices = anchor_center_index + sampled_offsets

    if closed:
        candidate_indices = np.mod(candidate_indices, num_points)
    else:
        candidate_indices = np.clip(candidate_indices, 0, num_points - 1)

        # Safety: for open curves, clipping might bring a negative back to anchor.
        # If that happens, push it by min_offset when possible.
        for i in range(len(candidate_indices)):
            if candidate_indices[i] == anchor_center_index:
                if anchor_center_index + min_offset < num_points:
                    candidate_indices[i] = anchor_center_index + min_offset
                elif anchor_center_index - min_offset >= 0:
                    candidate_indices[i] = anchor_center_index - min_offset
                else:
                    raise ValueError("Could not find a valid negative index for open curve.")

    return candidate_indices.astype(np.int64)


def build_tangent_training_tuple(
    curve_points: Array,
    family: str,
    anchor_center_index: int,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    *,
    closed: bool = True,
    patch_mode: str = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
    transform_kwargs: dict[str, Any] | None = None,
) -> TangentTrainingTuple:
    """
    Build one training tuple for a chosen transformation family.

    Version 1 design:
        - anchor    : patch at center p on original curve
        - positive  : patch at same center p on transformed curve
        - negatives : nearby patches q != p on the same transformed curve

    This keeps the transformation family consistent across positive and negatives.

    Args:
        curve_points:
            Dense sampled curve, shape (N, 2).
        family:
            Transformation family, e.g.:
                "euclidean", "similarity", "equi_affine", "affine"
        anchor_center_index:
            Fixed anchor center index p.
        patch_size:
            Odd patch size >= 3.
        half_width:
            Patch neighborhood radius in index units.
        num_negatives:
            Number of negative patches.
        negative_min_offset:
            Minimum absolute offset from anchor center for negatives.
        negative_max_offset:
            Maximum absolute offset from anchor center for negatives.
        closed:
            Whether the dense curve should wrap around cyclically.
        patch_mode:
            Patch sampling mode.
        jitter_fraction:
            Jitter strength for patch sampling.
        rng:
            Optional NumPy random generator.
        transform_kwargs:
            Extra keyword arguments forwarded to sample_transformation().

    Returns:
        TangentTrainingTuple
    """
    rng = _ensure_rng(rng)

    curve_points = np.asarray(curve_points, dtype=np.float64)
    if curve_points.ndim != 2 or curve_points.shape[1] != 2:
        raise ValueError("curve_points must have shape (N, 2).")

    num_points = len(curve_points)
    if not (0 <= anchor_center_index < num_points):
        raise ValueError("anchor_center_index is out of range.")

    if transform_kwargs is None:
        transform_kwargs = {}

    # 1) Anchor on original curve
    anchor = sample_patch_around_index(
        curve_points=curve_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    # 2) Sample transformation from requested family
    transform = sample_transformation(
        family=family,
        rng=rng,
        **transform_kwargs,
    )

    transformed_curve = apply_transformation(curve_points, transform)

    # 3) Positive: same center index, transformed curve
    positive = sample_patch_around_index(
        curve_points=transformed_curve,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    # 4) Negatives: nearby but different centers on same transformed curve
    negative_center_indices = _sample_local_negative_indices(
        num_points=num_points,
        anchor_center_index=anchor_center_index,
        num_negatives=num_negatives,
        min_offset=negative_min_offset,
        max_offset=negative_max_offset,
        closed=closed,
        rng=rng,
    )

    negatives: list[CurvePatchSample] = []
    for neg_idx in negative_center_indices:
        neg_patch = sample_patch_around_index(
            curve_points=transformed_curve,
            center_index=int(neg_idx),
            patch_size=patch_size,
            half_width=half_width,
            mode=patch_mode,
            closed=closed,
            rng=rng,
            jitter_fraction=jitter_fraction,
        )
        negatives.append(neg_patch)

    return TangentTrainingTuple(
        family=family,
        anchor=anchor,
        positive=positive,
        negatives=negatives,
        transform=transform,
        anchor_center_index=int(anchor_center_index),
        negative_center_indices=negative_center_indices,
    )


def build_random_tangent_training_tuple(
    curve_points: Array,
    family: str,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    *,
    closed: bool = True,
    patch_mode: str = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    rng: np.random.Generator | None = None,
    transform_kwargs: dict[str, Any] | None = None,
    valid_center_margin: int | None = None,
) -> TangentTrainingTuple:
    """
    Same as build_tangent_training_tuple(), but chooses the anchor center randomly.
    """
    rng = _ensure_rng(rng)

    curve_points = np.asarray(curve_points, dtype=np.float64)
    if curve_points.ndim != 2 or curve_points.shape[1] != 2:
        raise ValueError("curve_points must have shape (N, 2).")

    num_points = len(curve_points)

    if closed:
        anchor_center_index = int(rng.integers(0, num_points))
    else:
        if valid_center_margin is None:
            valid_center_margin = max(half_width, negative_max_offset)
        left = valid_center_margin
        right = num_points - valid_center_margin
        if left >= right:
            raise ValueError("No valid center indices remain for the requested margin.")
        anchor_center_index = int(rng.integers(left, right))

    return build_tangent_training_tuple(
        curve_points=curve_points,
        family=family,
        anchor_center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=negative_min_offset,
        negative_max_offset=negative_max_offset,
        closed=closed,
        patch_mode=patch_mode,
        jitter_fraction=jitter_fraction,
        rng=rng,
        transform_kwargs=transform_kwargs,
    )
