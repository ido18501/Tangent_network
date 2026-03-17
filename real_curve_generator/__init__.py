from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class RawCurve:
    points: np.ndarray
    closed: bool
    source: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalCurve:
    image_points: np.ndarray
    canonical_points: np.ndarray
    closed: bool
    normalization: dict[str, Any]


@dataclass
class RealCurveSample:
    dense_curve: np.ndarray
    image_curve: np.ndarray
    closed: bool
    score: float
    image_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
