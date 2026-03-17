from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ContourExtractionConfig:
    # preprocessing
    blur_kernel_size: int = 5
    canny_low: int = 80
    canny_high: int = 160

    # basic contour filters
    min_contour_points: int = 30
    min_arc_length: float = 40.0

    # simplification
    polygon_epsilon_frac: float = 0.002

    # bbox filtering
    min_bbox_width: float = 8.0
    min_bbox_height: float = 8.0

    # border filtering
    remove_border_touching: bool = False
    border_margin: int = 1

    # jaggedness (optional)
    max_mean_turn_angle_deg: float | None = None
    max_sharp_turn_fraction: float | None = None
    sharp_turn_threshold_deg: float = 75.0

    # final selection
    max_contours_per_image: int = 50


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def bgr_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def smooth_gray_image(gray: np.ndarray, blur_kernel_size: int = 5) -> np.ndarray:
    if blur_kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be odd.")
    return cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)


def detect_edges(gray_smoothed: np.ndarray, canny_low: int, canny_high: int) -> np.ndarray:
    return cv2.Canny(gray_smoothed, threshold1=canny_low, threshold2=canny_high)


def _contour_to_xy_array(contour: np.ndarray) -> np.ndarray:
    return contour.reshape(-1, 2).astype(np.float64)


def _arc_length(contour_xy: np.ndarray, closed: bool) -> float:
    if len(contour_xy) < 2:
        return 0.0

    diffs = np.diff(contour_xy, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())

    if closed and len(contour_xy) >= 3:
        length += float(np.linalg.norm(contour_xy[0] - contour_xy[-1]))

    return length


def simplify_contour(contour_xy: np.ndarray, epsilon_frac: float, closed: bool) -> np.ndarray:
    if epsilon_frac <= 0.0 or len(contour_xy) < 3:
        return contour_xy

    arc_len = _arc_length(contour_xy, closed=closed)
    epsilon = epsilon_frac * arc_len

    contour_cv = contour_xy.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour_cv, epsilon, closed)

    return approx.reshape(-1, 2).astype(np.float64)


def _estimate_closed(contour_xy: np.ndarray, tol: float = 1.5) -> bool:
    if len(contour_xy) < 3:
        return False
    return np.linalg.norm(contour_xy[0] - contour_xy[-1]) <= tol


def _bbox_stats(contour_xy: np.ndarray) -> tuple[float, float]:
    x_min = float(np.min(contour_xy[:, 0]))
    x_max = float(np.max(contour_xy[:, 0]))
    y_min = float(np.min(contour_xy[:, 1]))
    y_max = float(np.max(contour_xy[:, 1]))
    return x_max - x_min, y_max - y_min


def _touches_border(
    contour_xy: np.ndarray,
    image_shape: tuple[int, int],
    margin: int,
) -> bool:
    h, w = image_shape
    x = contour_xy[:, 0]
    y = contour_xy[:, 1]
    return bool(
        np.any(x <= margin)
        or np.any(x >= (w - 1 - margin))
        or np.any(y <= margin)
        or np.any(y >= (h - 1 - margin))
    )


def _turn_angle_stats(contour_xy: np.ndarray, sharp_turn_threshold_deg: float = 75.0) -> tuple[float, float]:
    """
    Returns:
        mean_turn_angle_deg,
        sharp_turn_fraction
    """
    if len(contour_xy) < 3:
        return 180.0, 1.0

    v_prev = contour_xy[1:-1] - contour_xy[:-2]
    v_next = contour_xy[2:] - contour_xy[1:-1]

    n1 = np.linalg.norm(v_prev, axis=1)
    n2 = np.linalg.norm(v_next, axis=1)

    valid = (n1 > 1e-8) & (n2 > 1e-8)
    if not np.any(valid):
        return 180.0, 1.0

    v_prev = v_prev[valid] / n1[valid][:, None]
    v_next = v_next[valid] / n2[valid][:, None]

    cosang = np.sum(v_prev * v_next, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosang))

    mean_turn = float(np.mean(angles))
    sharp_fraction = float(np.mean(angles >= sharp_turn_threshold_deg))
    return mean_turn, sharp_fraction


def _quality_score(
    contour_xy: np.ndarray,
    closed: bool,
    image_shape: tuple[int, int],
    config: ContourExtractionConfig,
) -> float:
    arc_len = _arc_length(contour_xy, closed=closed)
    bbox_w, bbox_h = _bbox_stats(contour_xy)
    mean_turn, sharp_turn_fraction = _turn_angle_stats(
        contour_xy,
        sharp_turn_threshold_deg=config.sharp_turn_threshold_deg,
    )

    # Favor long and spatially extended contours.
    # Penalize jaggedness only mildly.
    score = (
        1.0 * arc_len
        + 0.8 * max(bbox_w, bbox_h)
        + 0.3 * min(bbox_w, bbox_h)
        - 0.5 * mean_turn
        - 25.0 * sharp_turn_fraction
    )

    if config.remove_border_touching and _touches_border(
        contour_xy, image_shape=image_shape, margin=config.border_margin
    ):
        score -= 50.0

    return float(score)


def extract_contours(
    image_bgr: np.ndarray,
    config: ContourExtractionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    gray = bgr_to_gray(image_bgr)
    gray_smoothed = smooth_gray_image(gray, blur_kernel_size=config.blur_kernel_size)
    edges = detect_edges(gray_smoothed, config.canny_low, config.canny_high)

    raw_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    image_shape = gray.shape[:2]
    kept: list[tuple[float, np.ndarray]] = []

    for contour in raw_contours:
        contour_xy = _contour_to_xy_array(contour)

        if contour_xy.shape[0] < config.min_contour_points:
            continue

        closed = _estimate_closed(contour_xy)

        contour_xy = simplify_contour(
            contour_xy,
            epsilon_frac=config.polygon_epsilon_frac,
            closed=closed,
        )

        if contour_xy.shape[0] < config.min_contour_points:
            continue

        arc_len = _arc_length(contour_xy, closed=closed)
        if arc_len < config.min_arc_length:
            continue

        bbox_w, bbox_h = _bbox_stats(contour_xy)
        if bbox_w < config.min_bbox_width or bbox_h < config.min_bbox_height:
            continue

        if config.remove_border_touching and _touches_border(
            contour_xy,
            image_shape=image_shape,
            margin=config.border_margin,
        ):
            continue

        mean_turn, sharp_turn_fraction = _turn_angle_stats(
            contour_xy,
            sharp_turn_threshold_deg=config.sharp_turn_threshold_deg,
        )

        if config.max_mean_turn_angle_deg is not None and mean_turn > config.max_mean_turn_angle_deg:
            continue

        if config.max_sharp_turn_fraction is not None and sharp_turn_fraction > config.max_sharp_turn_fraction:
            continue

        score = _quality_score(
            contour_xy,
            closed=closed,
            image_shape=image_shape,
            config=config,
        )
        kept.append((score, contour_xy))

    kept.sort(key=lambda x: x[0], reverse=True)

    if config.max_contours_per_image is not None:
        kept = kept[: config.max_contours_per_image]

    contours_xy = [c.astype(np.float32) for _, c in kept]
    return gray, edges, contours_xy


def draw_contours_on_image(
    image_bgr: np.ndarray,
    contours_xy: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    draw_points: bool = False,
) -> np.ndarray:
    canvas = image_bgr.copy()

    for contour_xy in contours_xy:
        if len(contour_xy) < 2:
            continue

        pts = np.round(contour_xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)

        if draw_points:
            for x, y in contour_xy:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)

    return canvas
