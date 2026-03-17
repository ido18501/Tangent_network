from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, restoration, segmentation, util

from . import RawCurve


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _preprocess_for_geometry(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = _as_rgb(_normalize_image(image))
    # Edge-preserving denoise to suppress texture while preserving boundaries.
    denoised = restoration.denoise_bilateral(
        rgb,
        sigma_color=0.08,
        sigma_spatial=3,
        channel_axis=-1,
    )
    lab = color.rgb2lab(denoised)
    gray = color.rgb2gray(denoised)
    return denoised, lab, gray


def _ordered_curve_from_contour(contour_rc: np.ndarray, closed: bool) -> np.ndarray:
    # skimage returns (row, col); convert to (x, y)
    pts = np.stack([contour_rc[:, 1], contour_rc[:, 0]], axis=1).astype(np.float32)
    if closed and len(pts) > 2:
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.vstack([pts, pts[:1]])
    return pts


def _segmentation_region_boundaries(
    image_rgb: np.ndarray,
    lab: np.ndarray,
    gray: np.ndarray,
) -> list[RawCurve]:
    curves: list[RawCurve] = []

    # Felzenszwalb is much more stable than raw edges for suppressing texture.
    seg = segmentation.felzenszwalb(
        util.img_as_float(image_rgb),
        scale=130,
        sigma=0.8,
        min_size=160,
    )

    grad = filters.sobel(gray)
    h, w = gray.shape
    image_diag = float(np.hypot(h, w))

    labels, counts = np.unique(seg, return_counts=True)
    for label, area in zip(labels, counts):
        if area < 140:
            continue
        mask = seg == label
        if mask.mean() > 0.6:
            continue

        mask = morphology.opening(mask, morphology.disk(1))
        mask = morphology.closing(mask, morphology.disk(2))
        mask = ndi.binary_fill_holes(mask)
        if mask.sum() < 140:
            continue

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours_cv, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_cv:
            continue

        for contour_cv in contours_cv:
            contour_cv = contour_cv[:, 0, :]  # (N,2) in (x,y)
            if len(contour_cv) < 50:
                continue

            pts = contour_cv.astype(np.float32)

            if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
                pts = np.vstack([pts, pts[:1]])

            min_xy = pts.min(axis=0)
            max_xy = pts.max(axis=0)
            bbox_diag = float(np.linalg.norm(max_xy - min_xy))
            if bbox_diag < 0.10 * image_diag:
                continue

            rr = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            cc = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            contrast = float(np.mean(grad[rr, cc]))

            # texture suppression: reject regions whose interior is too textured
            ys, xs = np.nonzero(mask)
            interior_std = float(np.std(gray[ys, xs])) if len(xs) > 0 else 0.0

            confidence = float(0.72 + min(0.28, 3.5 * contrast))
            curves.append(
                RawCurve(
                    points=pts,
                    closed=True,
                    source="region_boundary",
                    confidence=confidence,
                    metadata={
                        "region_area": int(mask.sum()),
                        "bbox_diag": bbox_diag,
                        "contrast_support": contrast,
                        "interior_std": interior_std,
                    },
                )
            )
    return curves


def _strong_object_edges(gray: np.ndarray) -> list[RawCurve]:
    curves: list[RawCurve] = []

    gmag = filters.scharr(gray)
    smooth_gmag = filters.gaussian(gmag, sigma=1.0)
    hi = np.percentile(smooth_gmag, 88)
    if hi <= 0:
        return curves

    # High-gradient ridges; clean before tracing.
    strong = smooth_gmag > hi
    strong = morphology.closing(strong, morphology.disk(1))
    labeled, num = ndi.label(strong)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        keep = counts >= 48
        keep[0] = False
        strong = keep[labeled]
    strong = morphology.skeletonize(strong)

    contours = measure.find_contours(strong.astype(np.uint8), 0.5)
    h, w = gray.shape
    image_diag = float(np.hypot(h, w))
    for contour in contours:
        if len(contour) < 30:
            continue
        pts = _ordered_curve_from_contour(contour, closed=False)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        bbox_diag = float(np.linalg.norm(max_xy - min_xy))
        if bbox_diag < 0.06 * image_diag:
            continue
        rr = np.clip(np.round(contour[:, 0]).astype(int), 0, h - 1)
        cc = np.clip(np.round(contour[:, 1]).astype(int), 0, w - 1)
        contrast = float(np.mean(smooth_gmag[rr, cc]))
        curves.append(
            RawCurve(
                points=pts,
                closed=False,
                source="strong_edge_skeleton",
                confidence=float(0.35 + min(0.45, 4.0 * contrast)),
                metadata={
                    "bbox_diag": bbox_diag,
                    "contrast_support": contrast,
                },
            )
        )
    return curves


def _deduplicate_curves(curves: Iterable[RawCurve]) -> list[RawCurve]:
    deduped: list[RawCurve] = []
    signatures: set[tuple[int, int, int, int, bool, str]] = set()
    for curve in curves:
        pts = curve.points
        min_xy = np.floor(pts.min(axis=0) / 8).astype(int)
        max_xy = np.floor(pts.max(axis=0) / 8).astype(int)
        sig = (
            int(min_xy[0]),
            int(min_xy[1]),
            int(max_xy[0]),
            int(max_xy[1]),
            curve.closed,
            curve.source,
        )
        if sig in signatures:
            continue
        signatures.add(sig)
        deduped.append(curve)
    return deduped


def extract_curve_candidates(image: np.ndarray) -> list[RawCurve]:
    """
    Extract geometrically meaningful ordered curve candidates from a real image.

    The extractor deliberately prioritizes region boundaries and cleaned structural
    edges over raw texture-rich edge maps.
    """
    image_rgb, lab, gray = _preprocess_for_geometry(image)
    candidates: list[RawCurve] = []
    candidates.extend(_segmentation_region_boundaries(image_rgb, lab, gray))

    edge_curves = _strong_object_edges(gray)
    for c in edge_curves:
        c.confidence *= 0.72
        c.metadata["edge_fallback"] = True
    candidates.extend(edge_curves)

    # Sort by internal confidence before later filtering/ranking.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_curves(candidates)
