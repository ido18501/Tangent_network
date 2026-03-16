from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ContourExtractionConfig:
    blur_kernel_size: int = 5
    canny_low: int = 80
    canny_high: int = 160
    min_contour_points: int = 30
    min_arc_length: float = 40.0
    polygon_epsilon_frac: float = 0.0  # 0 means no simplification


def load_image(image_path: str) -> np.ndarray:
    """
    Load image as BGR uint8 using OpenCV.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def bgr_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale uint8.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def smooth_gray_image(gray: np.ndarray, blur_kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to suppress noise and texture.
    Kernel size must be odd.
    """
    if blur_kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be odd.")
    return cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)


def detect_edges(gray_smoothed: np.ndarray, canny_low: int, canny_high: int) -> np.ndarray:
    """
    Compute binary edge map using Canny.
    """
    edges = cv2.Canny(gray_smoothed, threshold1=canny_low, threshold2=canny_high)
    return edges


def _contour_to_xy_array(contour: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV contour of shape (N,1,2) to float32 array of shape (N,2),
    with columns (x, y).
    """
    contour_xy = contour.reshape(-1, 2).astype(np.float32)
    return contour_xy


def _arc_length(contour_xy: np.ndarray, closed: bool) -> float:
    """
    Compute polyline arc length.
    """
    if len(contour_xy) < 2:
        return 0.0

    diffs = np.diff(contour_xy, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())

    if closed and len(contour_xy) >= 3:
        length += float(np.linalg.norm(contour_xy[0] - contour_xy[-1]))

    return length


def simplify_contour(contour_xy: np.ndarray, epsilon_frac: float, closed: bool) -> np.ndarray:
    """
    Optionally simplify contour using approxPolyDP.
    """
    if epsilon_frac <= 0.0 or len(contour_xy) < 3:
        return contour_xy

    arc_len = _arc_length(contour_xy, closed=closed)
    epsilon = epsilon_frac * arc_len
    approx = cv2.approxPolyDP(contour_xy.reshape(-1, 1, 2), epsilon, closed)
    return approx.reshape(-1, 2).astype(np.float32)


def extract_contours(
    image_bgr: np.ndarray,
    config: ContourExtractionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Full image -> contours pipeline.

    Returns:
        gray: grayscale image
        edges: binary edge map
        contours_xy: list of contours, each shape (N_i, 2) in (x, y) format
    """
    gray = bgr_to_gray(image_bgr)
    gray_smoothed = smooth_gray_image(gray, blur_kernel_size=config.blur_kernel_size)
    edges = detect_edges(gray_smoothed, config.canny_low, config.canny_high)

    raw_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours_xy: List[np.ndarray] = []

    for contour in raw_contours:
        contour_xy = _contour_to_xy_array(contour)

        if contour_xy.shape[0] < config.min_contour_points:
            continue

        closed = False
        if contour_xy.shape[0] >= 3:
            closed = np.linalg.norm(contour_xy[0] - contour_xy[-1]) <= 1.5

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

        contours_xy.append(contour_xy)

    return gray, edges, contours_xy


def draw_contours_on_image(
    image_bgr: np.ndarray,
    contours_xy: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    draw_points: bool = False,
) -> np.ndarray:
    """
    Draw extracted contours on a copy of the image.
    """
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
