from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best = None
    best_area = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        return None

    return best.reshape(-1, 2).astype(np.float64)


def smooth_contour(points: np.ndarray, window: int = 9, passes: int = 2) -> np.ndarray:
    pts = points.astype(np.float64).copy()
    if len(pts) < window:
        return pts

    kernel = np.ones(window, dtype=np.float64) / window

    for _ in range(passes):
        x = np.pad(pts[:, 0], (window // 2, window // 2), mode="wrap")
        y = np.pad(pts[:, 1], (window // 2, window // 2), mode="wrap")
        x = np.convolve(x, kernel, mode="valid")
        y = np.convolve(y, kernel, mode="valid")
        pts = np.stack([x, y], axis=1)

    return pts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/grabcut_debug")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # center-biased rectangle
    rect = (
        int(0.12 * w),
        int(0.08 * h),
        int(0.76 * w),
        int(0.82 * h),
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contour = largest_external_contour(fg_mask)

    cv2.imwrite(str(out_dir / "01_original.png"), image)
    cv2.imwrite(str(out_dir / "02_mask.png"), fg_mask)

    overlay = image.copy()

    if contour is not None and len(contour) >= 20:
        contour = smooth_contour(contour, window=9, passes=2)
        pts = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        print(f"Found contour with {len(contour)} points")
    else:
        print("No usable contour found")

    cv2.imwrite(str(out_dir / "03_overlay.png"), overlay)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
