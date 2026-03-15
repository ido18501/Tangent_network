from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from datasets.tangent_dataset import TangentSampleTensors


@dataclass
class TangentBatch:
    """
    Batched tensors for training.

    Shapes:
        anchor:    (B, patch_size, 2)
        positive:  (B, patch_size, 2)
        negatives: (B, num_negatives, patch_size, 2)
    """
    anchor: torch.Tensor
    positive: torch.Tensor
    negatives: torch.Tensor
    family: list[str]
    anchor_center_index: torch.Tensor
    negative_center_indices: torch.Tensor


def tangent_collate_fn(batch: Sequence[TangentSampleTensors]) -> TangentBatch:
    """
    Collate a list of TangentSampleTensors into one batch.

    Args:
        batch:
            Sequence of TangentSampleTensors.

    Returns:
        TangentBatch
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    anchor = torch.stack([sample.anchor for sample in batch], dim=0)
    positive = torch.stack([sample.positive for sample in batch], dim=0)
    negatives = torch.stack([sample.negatives for sample in batch], dim=0)

    family = [sample.family for sample in batch]
    anchor_center_index = torch.tensor(
        [sample.anchor_center_index for sample in batch],
        dtype=torch.long,
    )
    negative_center_indices = torch.stack(
        [sample.negative_center_indices for sample in batch],
        dim=0,
    )

    return TangentBatch(
        anchor=anchor,
        positive=positive,
        negatives=negatives,
        family=family,
        anchor_center_index=anchor_center_index,
        negative_center_indices=negative_center_indices,
    )
