from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from utils.patch_sampling import CurvePatchSample, sample_patch_around_index
from utils.transformations import Transformation2D, apply_transformation, sample_transformation


Array = np.ndarray
PatchSamplingMode = Literal["uniform_symmetric", "jittered_symmetric"]


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class RealCurveExtractionConfig:
    """
    Stronger real-image curve extraction config.

    Strategy:
      1) Prefer region boundaries from thresholded masks
      2) Add region boundaries from color quantization
      3) Add edge-based contours only as fallback
      4) Filter + rank + deduplicate candidates
    """
    # global filtering
    min_contour_points: int = 40
    min_arc_length: float = 80.0
    min_bbox_diag_frac: float = 0.06
    max_candidates_per_image: int = 40
    dedup_center_dist_frac: float = 0.04
    dedup_length_rel_tol: float = 0.15

    # closed / open detection
    closed_endpoint_tol: float = 4.0

    # smoothing / simplification
    gaussian_blur_ksize: int = 7
    gaussian_blur_sigma: float = 1.4
    simplify_epsilon_frac: float = 0.003
    contour_smooth_window: int = 7
    contour_smooth_passes: int = 2

    # region-threshold extraction
    enable_threshold_regions: bool = True
    num_threshold_levels: int = 7
    threshold_min_component_area_frac: float = 0.0015
    threshold_morph_kernel: int = 5

    # color-quantized region extraction
    enable_color_regions: bool = True
    color_quantization_k: int = 5
    color_min_component_area_frac: float = 0.0025
    color_morph_kernel: int = 5

    # edge extraction fallback
    enable_edges: bool = True
    canny_low: int = 80
    canny_high: int = 160
    edge_morph_kernel: int = 3

    # candidate quality scoring
    min_gradient_support: float = 0.18
    max_roughness: float = 1.75

    # canonicalization
    canonical_dense_num_points: int = 300
    target_extent: float = 0.6

    # patch batch center filtering
    valid_center_margin: int | None = None


@dataclass
class RawCurveCandidate:
    image_points: Array              # shape (N,2), ordered image-space points
    closed: bool
    source: str                      # e.g. threshold_region / color_region / edge
    source_priority: float
    arc_length: float
    bbox: tuple[float, float, float, float]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurveNormalization:
    center: Array                    # shape (2,)
    scale: float
    target_extent: float


@dataclass
class CanonicalCurve:
    image_points: Array              # dense image-space curve, shape (M,2)
    canonical_points: Array          # dense normalized curve, shape (M,2)
    closed: bool
    normalization: CurveNormalization
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferencePatchBatch:
    patches: Array                   # (K, patch_size, 2)
    center_indices: Array            # (K,)
    closed: bool
    source: str
    curve_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTrainingTuple:
    family: str
    anchor: CurvePatchSample
    positive: CurvePatchSample
    negatives: list[CurvePatchSample]
    transform: Transformation2D
    anchor_center_index: int
    negative_center_indices: Array
    closed: bool
    source: str
    curve_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================
# Basic helpers
# =========================================================

def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


def _arc_length(points: Array, closed: bool) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())
    if closed and len(points) >= 3:
        length += float(np.linalg.norm(points[0] - points[-1]))
    return length


def _bbox_xy(points: Array) -> tuple[float, float, float, float]:
    x0 = float(np.min(points[:, 0]))
    y0 = float(np.min(points[:, 1]))
    x1 = float(np.max(points[:, 0]))
    y1 = float(np.max(points[:, 1]))
    return (x0, y0, x1, y1)


def _bbox_diag(points: Array) -> float:
    x0, y0, x1, y1 = _bbox_xy(points)
    return float(np.hypot(x1 - x0, y1 - y0))


def _center_of_bbox(points: Array) -> Array:
    x0, y0, x1, y1 = _bbox_xy(points)
    return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)


def _contour_to_xy_array(contour: np.ndarray) -> Array:
    return contour.reshape(-1, 2).astype(np.float64)


def _is_closed_by_endpoints(points: Array, tol: float) -> bool:
    if len(points) < 3:
        return False
    return float(np.linalg.norm(points[0] - points[-1])) <= tol


def _close_if_needed(points: Array, closed: bool, tol: float) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if not closed:
        return points
    if len(points) < 3:
        return points
    if np.linalg.norm(points[0] - points[-1]) <= tol:
        return points
    return np.vstack([points, points[0]])


def _moving_average_periodic_1d(x: Array, window: int) -> Array:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="wrap")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def _moving_average_open_1d(x: Array, window: int) -> Array:
    if window <= 1:
        return x.copy()
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def smooth_contour(points: Array, closed: bool, window: int = 7, passes: int = 2) -> Array:
    pts = np.asarray(points, dtype=np.float64).copy()
    if len(pts) < max(5, window):
        return pts

    for _ in range(passes):
        if closed:
            x = _moving_average_periodic_1d(pts[:, 0], window)
            y = _moving_average_periodic_1d(pts[:, 1], window)
        else:
            x = _moving_average_open_1d(pts[:, 0], window)
            y = _moving_average_open_1d(pts[:, 1], window)
        pts = np.stack([x, y], axis=1)

    return pts


def simplify_contour(points: Array, epsilon_frac: float, closed: bool) -> Array:
    if epsilon_frac <= 0.0 or len(points) < 3:
        return np.asarray(points, dtype=np.float64)

    arc_len = _arc_length(points, closed=closed)
    eps = float(epsilon_frac) * arc_len
    approx = cv2.approxPolyDP(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        eps,
        closed,
    )
    return approx.reshape(-1, 2).astype(np.float64)


def polygon_signed_area(points: Array) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    pts = points
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def ensure_consistent_orientation(points: Array, closed: bool) -> Array:
    pts = np.asarray(points, dtype=np.float64)
    if closed and polygon_signed_area(pts) < 0:
        return pts[::-1].copy()
    return pts


def compute_polyline_arclength(points: Array, closed: bool) -> tuple[Array, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {points.shape}")
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")

    if closed:
        pts_ext = _close_if_needed(points, closed=True, tol=1e-6)
        segs = pts_ext[1:] - pts_ext[:-1]
        seg_lens = np.linalg.norm(segs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(seg_lens.sum())
        return s, total
    else:
        segs = points[1:] - points[:-1]
        seg_lens = np.linalg.norm(segs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = float(seg_lens.sum())
        return s, total


def resample_contour_uniform(points: Array, num_points: int, closed: bool) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("Need at least 2 points.")
    if num_points < 3:
        raise ValueError("num_points must be >= 3.")

    s, total = compute_polyline_arclength(points, closed=closed)
    if total <= 1e-12:
        raise ValueError("Contour has near-zero arc length.")

    if closed:
        pts_ext = _close_if_needed(points, closed=True, tol=1e-6)
        target_s = np.linspace(0.0, total, num_points, endpoint=False)
    else:
        pts_ext = points
        target_s = np.linspace(0.0, total, num_points, endpoint=True)

    x = np.interp(target_s, s, pts_ext[:, 0])
    y = np.interp(target_s, s, pts_ext[:, 1])
    return np.stack([x, y], axis=1)


def get_max_abs_extent(points: Array) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.max(np.abs(points)))


def normalize_curve_to_training_canvas(
    points: Array,
    target_extent: float = 0.6,
) -> tuple[Array, CurveNormalization]:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center.reshape(1, 2)
    extent = get_max_abs_extent(centered)
    if extent <= 1e-12:
        raise ValueError("Curve extent too small to normalize.")
    scale = float(target_extent) / extent
    canonical = centered * scale
    return canonical, CurveNormalization(
        center=center,
        scale=float(scale),
        target_extent=float(target_extent),
    )


def get_valid_center_indices(num_points: int, half_width: int, closed: bool) -> Array:
    if closed:
        return np.arange(num_points, dtype=np.int64)
    left = half_width
    right = num_points - half_width
    if left >= right:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(left, right, dtype=np.int64)


def _estimate_tangents(points: Array, closed: bool) -> Array:
    pts = np.asarray(points, dtype=np.float64)
    if closed:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        tang = next_pts - prev_pts
    else:
        tang = np.zeros_like(pts)
        tang[1:-1] = pts[2:] - pts[:-2]
        tang[0] = pts[1] - pts[0]
        tang[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.clip(norms, 1e-12, None)
    return tang


def _estimate_curvature_roughness(points: Array, closed: bool) -> float:
    """
    A simple roughness score: large means jagged / noisy.
    """
    tang = _estimate_tangents(points, closed=closed)
    if closed:
        dt = np.roll(tang, -1, axis=0) - tang
        turn = np.linalg.norm(dt, axis=1)
    else:
        if len(tang) < 3:
            return np.inf
        dt = tang[1:] - tang[:-1]
        turn = np.linalg.norm(dt, axis=1)

    if len(turn) == 0:
        return np.inf

    median_turn = float(np.median(turn))
    mean_turn = float(np.mean(turn))
    if median_turn <= 1e-12:
        return mean_turn / 1e-12
    return mean_turn / median_turn


def _curve_gradient_support(points: Array, grad_mag: Array) -> float:
    """
    Average normalized gradient magnitude sampled along contour points.
    """
    h, w = grad_mag.shape
    pts = np.asarray(points, dtype=np.float64)
    x = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)
    vals = grad_mag[y, x]

    gmax = float(np.max(grad_mag))
    if gmax <= 1e-12:
        return 0.0
    return float(np.mean(vals / gmax))


def _fill_ratio_for_closed_curve(points: Array) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    area = abs(polygon_signed_area(pts))
    x0, y0, x1, y1 = _bbox_xy(pts)
    box_area = max((x1 - x0) * (y1 - y0), 1e-12)
    return float(area / box_area)


# =========================================================
# Candidate extraction pieces
# =========================================================

def _prepare_gray_and_gradients(
    image_bgr: Array,
    config: RealCurveExtractionConfig,
) -> tuple[Array, Array]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (config.gaussian_blur_ksize, config.gaussian_blur_ksize),
        config.gaussian_blur_sigma,
    )
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return gray, grad_mag


def _binary_components_to_contours(
    mask: Array,
    min_component_area: int,
) -> list[np.ndarray]:
    """
    Convert connected components in a binary mask into external contours.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours: list[np.ndarray] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        component = np.zeros_like(mask)
        component[labels == label] = 255

        found, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in found:
            if c.shape[0] >= 3:
                contours.append(c)

    return contours


def _extract_threshold_region_candidates(
    image_bgr: Array,
    gray: Array,
    grad_mag: Array,
    config: RealCurveExtractionConfig,
) -> list[RawCurveCandidate]:
    h, w = gray.shape
    min_component_area = max(20, int(config.threshold_min_component_area_frac * h * w))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.threshold_morph_kernel, config.threshold_morph_kernel),
    )

    levels = np.linspace(25, 230, config.num_threshold_levels).astype(np.uint8)
    candidates: list[RawCurveCandidate] = []

    for thr in levels:
        _, mask_hi = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY)
        _, mask_lo = cv2.threshold(gray, int(thr), 255, cv2.THRESH_BINARY_INV)

        for mask in [mask_hi, mask_lo]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours = _binary_components_to_contours(mask, min_component_area=min_component_area)
            for contour in contours:
                pts = _contour_to_xy_array(contour)
                cand = _make_candidate_from_points(
                    pts,
                    closed=True,
                    source="threshold_region",
                    source_priority=3.0,
                    image_shape=(h, w),
                    grad_mag=grad_mag,
                    config=config,
                )
                if cand is not None:
                    candidates.append(cand)

    return candidates


def _extract_color_region_candidates(
    image_bgr: Array,
    grad_mag: Array,
    config: RealCurveExtractionConfig,
) -> list[RawCurveCandidate]:
    h, w = image_bgr.shape[:2]
    min_component_area = max(20, int(config.color_min_component_area_frac * h * w))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.color_morph_kernel, config.color_morph_kernel),
    )

    Z = image_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        25,
        1.0,
    )
    k = max(2, int(config.color_quantization_k))
    _compactness, labels, centers = cv2.kmeans(
        Z,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(h, w)

    candidates: list[RawCurveCandidate] = []
    for c in range(k):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[labels == c] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = _binary_components_to_contours(mask, min_component_area=min_component_area)
        for contour in contours:
            pts = _contour_to_xy_array(contour)
            cand = _make_candidate_from_points(
                pts,
                closed=True,
                source="color_region",
                source_priority=2.5,
                image_shape=(h, w),
                grad_mag=grad_mag,
                config=config,
            )
            if cand is not None:
                candidates.append(cand)

    return candidates


def _extract_edge_candidates(
    gray: Array,
    grad_mag: Array,
    config: RealCurveExtractionConfig,
) -> list[RawCurveCandidate]:
    h, w = gray.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.edge_morph_kernel, config.edge_morph_kernel),
    )

    edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    raw_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[RawCurveCandidate] = []

    for contour in raw_contours:
        pts = _contour_to_xy_array(contour)
        closed = _is_closed_by_endpoints(pts, tol=config.closed_endpoint_tol)
        cand = _make_candidate_from_points(
            pts,
            closed=closed,
            source="edge",
            source_priority=1.0,
            image_shape=(h, w),
            grad_mag=grad_mag,
            config=config,
        )
        if cand is not None:
            candidates.append(cand)

    return candidates


# =========================================================
# Candidate scoring / filtering
# =========================================================

def _make_candidate_from_points(
    points: Array,
    *,
    closed: bool,
    source: str,
    source_priority: float,
    image_shape: tuple[int, int],
    grad_mag: Array,
    config: RealCurveExtractionConfig,
) -> RawCurveCandidate | None:
    h, w = image_shape
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) < config.min_contour_points:
        return None

    pts = smooth_contour(
        pts,
        closed=closed,
        window=config.contour_smooth_window,
        passes=config.contour_smooth_passes,
    )
    pts = simplify_contour(
        pts,
        epsilon_frac=config.simplify_epsilon_frac,
        closed=closed,
    )
    pts = ensure_consistent_orientation(pts, closed=closed)

    if len(pts) < config.min_contour_points:
        return None

    arc_len = _arc_length(pts, closed=closed)
    if arc_len < config.min_arc_length:
        return None

    diag = _bbox_diag(pts)
    image_diag = float(np.hypot(w, h))
    if diag < config.min_bbox_diag_frac * image_diag:
        return None

    gradient_support = _curve_gradient_support(pts, grad_mag=grad_mag)
    if gradient_support < config.min_gradient_support:
        return None

    roughness = _estimate_curvature_roughness(pts, closed=closed)
    if not np.isfinite(roughness) or roughness > config.max_roughness:
        return None

    fill_ratio = _fill_ratio_for_closed_curve(pts) if closed else 0.0

    # score: prioritize source type, then meaningful size/support, then smoothness
    score = (
        2.0 * source_priority
        + 1.4 * gradient_support
        + 0.6 * min(diag / image_diag, 1.0)
        + 0.5 * min(arc_len / image_diag, 2.0)
        + (0.35 * fill_ratio if closed else 0.0)
        - 0.7 * roughness
    )

    return RawCurveCandidate(
        image_points=pts.astype(np.float64),
        closed=bool(closed),
        source=source,
        source_priority=float(source_priority),
        arc_length=float(arc_len),
        bbox=_bbox_xy(pts),
        score=float(score),
        metadata={
            "gradient_support": float(gradient_support),
            "roughness": float(roughness),
            "fill_ratio": float(fill_ratio),
            "bbox_diag": float(diag),
            "num_points": int(len(pts)),
        },
    )


def _are_duplicate_candidates(
    a: RawCurveCandidate,
    b: RawCurveCandidate,
    image_diag: float,
    config: RealCurveExtractionConfig,
) -> bool:
    if a.closed != b.closed:
        return False

    ca = _center_of_bbox(a.image_points)
    cb = _center_of_bbox(b.image_points)
    center_dist = float(np.linalg.norm(ca - cb))
    if center_dist > config.dedup_center_dist_frac * image_diag:
        return False

    len_a = max(a.arc_length, 1e-12)
    len_b = max(b.arc_length, 1e-12)
    rel = abs(len_a - len_b) / max(len_a, len_b)
    if rel > config.dedup_length_rel_tol:
        return False

    return True


def _deduplicate_candidates(
    candidates: list[RawCurveCandidate],
    image_shape: tuple[int, int],
    config: RealCurveExtractionConfig,
) -> list[RawCurveCandidate]:
    h, w = image_shape
    image_diag = float(np.hypot(w, h))

    kept: list[RawCurveCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = False
        for prev in kept:
            if _are_duplicate_candidates(cand, prev, image_diag=image_diag, config=config):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
        if len(kept) >= config.max_candidates_per_image:
            break
    return kept


# =========================================================
# Public extraction API
# =========================================================

def extract_curve_candidates_from_image(
    image: Array,
    config: RealCurveExtractionConfig,
) -> list[RawCurveCandidate]:
    """
    Main stronger extractor.

    Preference order:
      1) threshold-region boundaries
      2) color-region boundaries
      3) edges fallback
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in BGR format.")

    gray, grad_mag = _prepare_gray_and_gradients(image, config)
    h, w = gray.shape

    all_candidates: list[RawCurveCandidate] = []

    if config.enable_threshold_regions:
        all_candidates.extend(
            _extract_threshold_region_candidates(image, gray, grad_mag, config)
        )

    if config.enable_color_regions:
        all_candidates.extend(
            _extract_color_region_candidates(image, grad_mag, config)
        )

    if config.enable_edges:
        all_candidates.extend(
            _extract_edge_candidates(gray, grad_mag, config)
        )

    all_candidates = _deduplicate_candidates(
        all_candidates,
        image_shape=(h, w),
        config=config,
    )
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    return all_candidates


# =========================================================
# Canonicalization API
# =========================================================

def canonicalize_curve_candidate(
    candidate: RawCurveCandidate,
    config: RealCurveExtractionConfig,
) -> CanonicalCurve:
    dense_image = resample_contour_uniform(
        candidate.image_points,
        num_points=config.canonical_dense_num_points,
        closed=candidate.closed,
    )

    dense_canonical, norm = normalize_curve_to_training_canvas(
        dense_image,
        target_extent=config.target_extent,
    )

    return CanonicalCurve(
        image_points=dense_image,
        canonical_points=dense_canonical,
        closed=candidate.closed,
        normalization=norm,
        source=candidate.source,
        score=candidate.score,
        metadata=dict(candidate.metadata),
    )


# =========================================================
# Patch / tuple API
# =========================================================

def build_inference_patches(
    curve: CanonicalCurve,
    patch_size: int,
    half_width: int,
    patch_mode: PatchSamplingMode = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    center_indices: Array | None = None,
    stride: int = 1,
    rng: np.random.Generator | None = None,
) -> InferencePatchBatch:
    """
    Build model-ready patches from the dense canonical curve while preserving
    the same local support semantics as training.
    """
    rng = _ensure_rng(rng)
    num_points = len(curve.canonical_points)

    if center_indices is None:
        valid = get_valid_center_indices(
            num_points=num_points,
            half_width=half_width,
            closed=curve.closed,
        )
        center_indices = valid[::max(1, stride)]

    center_indices = np.asarray(center_indices, dtype=np.int64)
    if len(center_indices) == 0:
        raise ValueError("No valid center indices provided.")

    patches = []
    for cidx in center_indices:
        sample = sample_patch_around_index(
            curve_points=curve.canonical_points,
            center_index=int(cidx),
            patch_size=patch_size,
            half_width=half_width,
            mode=patch_mode,
            closed=curve.closed,
            rng=rng,
            jitter_fraction=jitter_fraction,
        )
        patches.append(sample.centered_points)

    patches_np = np.stack(patches, axis=0)
    return InferencePatchBatch(
        patches=patches_np,
        center_indices=center_indices,
        closed=curve.closed,
        source=curve.source,
        curve_score=curve.score,
        metadata=dict(curve.metadata),
    )


def _sample_local_negative_indices(
    num_points: int,
    anchor_center_index: int,
    num_negatives: int,
    min_offset: int,
    max_offset: int,
    closed: bool,
    rng: np.random.Generator,
) -> Array:
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
                    raise ValueError("Could not find valid negative index for open curve.")

    return candidate_indices.astype(np.int64)


def _sample_random_patch_from_curve(
    curve_points: Array,
    patch_size: int,
    half_width: int,
    *,
    closed: bool,
    patch_mode: str,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> CurvePatchSample:
    num_points = len(curve_points)

    if closed:
        center_index = int(rng.integers(0, num_points))
    else:
        left = half_width
        right = num_points - half_width
        if left >= right:
            raise ValueError("No valid open-curve center for external negative.")
        center_index = int(rng.integers(left, right))

    return sample_patch_around_index(
        curve_points=curve_points,
        center_index=center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )


def build_real_training_tuple(
    curve: CanonicalCurve,
    family: str,
    patch_size: int,
    half_width: int,
    num_negatives: int,
    negative_min_offset: int,
    negative_max_offset: int,
    patch_mode: PatchSamplingMode = "jittered_symmetric",
    jitter_fraction: float = 0.25,
    anchor_center_index: int | None = None,
    rng: np.random.Generator | None = None,
    transform_kwargs: dict[str, Any] | None = None,
    external_negative_curves: list[CanonicalCurve] | None = None,
    num_cross_curve_negatives: int = 0,
) -> RealTrainingTuple:
    """
    Mirrors the synthetic contract:

      - anchor: patch on original canonical dense curve
      - positive: patch at same center on transformed dense curve,
                  independently resampled
      - negatives: nearby local patches on transformed dense curve
                   + optional cross-curve negatives
    """
    rng = _ensure_rng(rng)
    if transform_kwargs is None:
        transform_kwargs = {}
    if external_negative_curves is None:
        external_negative_curves = []

    valid_centers = get_valid_center_indices(
        num_points=len(curve.canonical_points),
        half_width=half_width,
        closed=curve.closed,
    )
    if len(valid_centers) == 0:
        raise ValueError("No valid center indices for this curve.")

    if anchor_center_index is None:
        anchor_center_index = int(rng.choice(valid_centers))
    else:
        anchor_center_index = int(anchor_center_index)

    anchor = sample_patch_around_index(
        curve_points=curve.canonical_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=curve.closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    transform = sample_transformation(
        family=family,
        rng=rng,
        **transform_kwargs,
    )

    transformed_curve_points = apply_transformation(curve.canonical_points, transform)

    positive = sample_patch_around_index(
        curve_points=transformed_curve_points,
        center_index=anchor_center_index,
        patch_size=patch_size,
        half_width=half_width,
        mode=patch_mode,
        closed=curve.closed,
        rng=rng,
        jitter_fraction=jitter_fraction,
    )

    num_cross_curve_negatives = int(max(0, min(num_cross_curve_negatives, num_negatives)))
    num_same_curve_negatives = int(num_negatives - num_cross_curve_negatives)

    negatives: list[CurvePatchSample] = []
    negative_center_indices_parts: list[Array] = []

    if num_same_curve_negatives > 0:
        same_curve_negative_indices = _sample_local_negative_indices(
            num_points=len(curve.canonical_points),
            anchor_center_index=anchor_center_index,
            num_negatives=num_same_curve_negatives,
            min_offset=negative_min_offset,
            max_offset=negative_max_offset,
            closed=curve.closed,
            rng=rng,
        )

        for neg_idx in same_curve_negative_indices:
            neg_patch = sample_patch_around_index(
                curve_points=transformed_curve_points,
                center_index=int(neg_idx),
                patch_size=patch_size,
                half_width=half_width,
                mode=patch_mode,
                closed=curve.closed,
                rng=rng,
                jitter_fraction=jitter_fraction,
            )
            negatives.append(neg_patch)

        negative_center_indices_parts.append(same_curve_negative_indices.astype(np.int64))

    if num_cross_curve_negatives > 0:
        if len(external_negative_curves) < num_cross_curve_negatives:
            raise ValueError(
                "Need at least num_cross_curve_negatives external curves, "
                f"got {len(external_negative_curves)}."
            )

        chosen = rng.choice(
            len(external_negative_curves),
            size=num_cross_curve_negatives,
            replace=False,
        )

        cross_curve_center_indices = np.full(num_cross_curve_negatives, -1, dtype=np.int64)

        for idx in chosen:
            ext_curve = external_negative_curves[int(idx)]
            ext_transformed = apply_transformation(ext_curve.canonical_points, transform)

            neg_patch = _sample_random_patch_from_curve(
                curve_points=ext_transformed,
                patch_size=patch_size,
                half_width=half_width,
                closed=ext_curve.closed,
                patch_mode=patch_mode,
                jitter_fraction=jitter_fraction,
                rng=rng,
            )
            negatives.append(neg_patch)

        negative_center_indices_parts.append(cross_curve_center_indices)

    if len(negatives) != num_negatives:
        raise RuntimeError(f"Expected {num_negatives} negatives, got {len(negatives)}.")

    negative_center_indices = np.concatenate(negative_center_indices_parts, axis=0)

    return RealTrainingTuple(
        family=family,
        anchor=anchor,
        positive=positive,
        negatives=negatives,
        transform=transform,
        anchor_center_index=int(anchor_center_index),
        negative_center_indices=negative_center_indices,
        closed=curve.closed,
        source=curve.source,
        curve_score=curve.score,
        metadata=dict(curve.metadata),
    )


# =========================================================
# Small optional utility for visualization
# =========================================================

def draw_curve_candidates_on_image(
    image_bgr: Array,
    candidates: list[RawCurveCandidate],
    top_k: int = 12,
    draw_points: bool = False,
) -> Array:
    canvas = image_bgr.copy()
    top = candidates[:top_k]

    colors = {
        "threshold_region": (0, 255, 0),
        "color_region": (255, 180, 0),
        "edge": (0, 0, 255),
    }

    for i, cand in enumerate(top):
        pts = np.round(cand.image_points).astype(np.int32).reshape(-1, 1, 2)
        color = colors.get(cand.source, (255, 255, 255))
        cv2.polylines(canvas, [pts], isClosed=cand.closed, color=color, thickness=2)

        if draw_points:
            for x, y in cand.image_points:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (255, 255, 255), -1)

        label = f"{i}: {cand.source} s={cand.score:.2f}"
        p = cand.image_points[0]
        tx = int(np.clip(round(p[0]), 0, canvas.shape[1] - 1))
        ty = int(np.clip(round(p[1]), 0, canvas.shape[0] - 1))
        cv2.putText(
            canvas,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas