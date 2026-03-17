from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import RealCurveSample
from .canonicalization import canonicalize_curve
from .extraction import extract_curve_candidates
from .filtering import filter_curves


def process_image_to_samples(
    image: np.ndarray,
    image_id: str,
    *,
    dense_points: int = 512,
    top_k: int = 12,
) -> list[RealCurveSample]:
    raw_curves = extract_curve_candidates(image)
    ranked = filter_curves(raw_curves, image, top_k=top_k)

    samples: list[RealCurveSample] = []
    for rank, (curve, score) in enumerate(ranked):
        canonical = canonicalize_curve(curve, dense_points=dense_points)
        metadata = dict(curve.metadata)
        metadata.update(
            {
                "source": curve.source,
                "confidence": float(curve.confidence),
                "rank": rank,
                "normalization": canonical.normalization,
            }
        )
        samples.append(
            RealCurveSample(
                dense_curve=canonical.canonical_points.astype(np.float32),
                image_curve=canonical.image_points.astype(np.float32),
                closed=canonical.closed,
                score=float(score),
                image_id=image_id,
                metadata=metadata,
            )
        )
    return samples


def save_samples_npz(samples: list[RealCurveSample], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "num_curves": np.int32(len(samples)),
    }
    for i, sample in enumerate(samples):
        payload[f"curve_{i}"] = sample.dense_curve.astype(np.float32)
        payload[f"image_curve_{i}"] = sample.image_curve.astype(np.float32)
        payload[f"closed_{i}"] = np.bool_(sample.closed)
        payload[f"score_{i}"] = np.float32(sample.score)
        payload[f"metadata_{i}"] = np.array(sample.metadata, dtype=object)
        payload[f"image_id_{i}"] = np.array(sample.image_id)
    np.savez_compressed(output_path, **payload)


def load_samples_npz(path: str | Path) -> list[RealCurveSample]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    num_curves = int(data["num_curves"])
    samples: list[RealCurveSample] = []
    for i in range(num_curves):
        metadata = data[f"metadata_{i}"].item()
        samples.append(
            RealCurveSample(
                dense_curve=data[f"curve_{i}"].astype(np.float32),
                image_curve=data[f"image_curve_{i}"].astype(np.float32),
                closed=bool(data[f"closed_{i}"]),
                score=float(data[f"score_{i}"]),
                image_id=str(data[f"image_id_{i}"]),
                metadata=metadata,
            )
        )
    return samples


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
