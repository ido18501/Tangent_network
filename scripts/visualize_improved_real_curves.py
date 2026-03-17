from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.io_utils import list_image_files
from src.real_curve_generator import (
    RealCurveExtractionConfig,
    extract_curve_candidates_from_image,
)


def load_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def draw_curve_candidates_on_image(
    image_rgb: np.ndarray,
    candidates,
    *,
    top_k: int | None = None,
    draw_points: bool = False,
) -> np.ndarray:
    """
    Draw candidate curves on top of the RGB image.
    """
    canvas = image_rgb.copy()

    colors = {
        "threshold_region": np.array([0, 255, 0], dtype=np.uint8),   # green
        "color_region": np.array([255, 180, 0], dtype=np.uint8),     # orange
        "edge": np.array([255, 0, 0], dtype=np.uint8),               # red
    }

    draw_list = candidates if top_k is None else candidates[:top_k]

    h, w = canvas.shape[:2]

    for i, cand in enumerate(draw_list):
        pts = np.round(cand.image_points).astype(np.int64)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        color = colors.get(cand.source, np.array([255, 255, 255], dtype=np.uint8))

        # draw polyline by rasterizing points
        for j in range(len(pts) - 1):
            x0, y0 = pts[j]
            x1, y1 = pts[j + 1]

            rr = np.linspace(y0, y1, num=max(abs(y1 - y0), abs(x1 - x0)) + 1)
            cc = np.linspace(x0, x1, num=max(abs(y1 - y0), abs(x1 - x0)) + 1)

            rr = np.clip(np.round(rr).astype(np.int64), 0, h - 1)
            cc = np.clip(np.round(cc).astype(np.int64), 0, w - 1)
            canvas[rr, cc] = color

        if cand.closed and len(pts) >= 2:
            x0, y0 = pts[-1]
            x1, y1 = pts[0]
            rr = np.linspace(y0, y1, num=max(abs(y1 - y0), abs(x1 - x0)) + 1)
            cc = np.linspace(x0, x1, num=max(abs(y1 - y0), abs(x1 - x0)) + 1)
            rr = np.clip(np.round(rr).astype(np.int64), 0, h - 1)
            cc = np.clip(np.round(cc).astype(np.int64), 0, w - 1)
            canvas[rr, cc] = color

        if draw_points:
            for x, y in pts:
                canvas[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = color

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images" / "BSR" / "BSDS500" / "data" / "images" / "train"),
    )
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=8)

    # improved extractor config
    parser.add_argument("--min_contour_points", type=int, default=40)
    parser.add_argument("--min_arc_length", type=float, default=80.0)
    parser.add_argument("--min_bbox_diag_frac", type=float, default=0.06)
    parser.add_argument("--max_candidates_per_image", type=int, default=40)

    parser.add_argument("--gaussian_blur_ksize", type=int, default=7)
    parser.add_argument("--gaussian_blur_sigma", type=float, default=1.4)
    parser.add_argument("--simplify_epsilon_frac", type=float, default=0.003)
    parser.add_argument("--contour_smooth_window", type=int, default=7)
    parser.add_argument("--contour_smooth_passes", type=int, default=2)

    parser.add_argument("--num_threshold_levels", type=int, default=7)
    parser.add_argument("--threshold_min_component_area_frac", type=float, default=0.0015)
    parser.add_argument("--threshold_morph_kernel", type=int, default=5)

    parser.add_argument("--color_quantization_k", type=int, default=5)
    parser.add_argument("--color_min_component_area_frac", type=float, default=0.0025)
    parser.add_argument("--color_morph_kernel", type=int, default=5)

    parser.add_argument("--canny_low", type=int, default=80)
    parser.add_argument("--canny_high", type=int, default=160)
    parser.add_argument("--edge_morph_kernel", type=int, default=3)

    parser.add_argument("--min_gradient_support", type=float, default=0.18)
    parser.add_argument("--max_roughness", type=float, default=1.75)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_files = list_image_files(input_dir)[:args.max_images]

    if not image_files:
        raise RuntimeError(f"No image files found in {input_dir}")

    config = RealCurveExtractionConfig(
        min_contour_points=args.min_contour_points,
        min_arc_length=args.min_arc_length,
        min_bbox_diag_frac=args.min_bbox_diag_frac,
        max_candidates_per_image=args.max_candidates_per_image,

        gaussian_blur_ksize=args.gaussian_blur_ksize,
        gaussian_blur_sigma=args.gaussian_blur_sigma,
        simplify_epsilon_frac=args.simplify_epsilon_frac,
        contour_smooth_window=args.contour_smooth_window,
        contour_smooth_passes=args.contour_smooth_passes,

        num_threshold_levels=args.num_threshold_levels,
        threshold_min_component_area_frac=args.threshold_min_component_area_frac,
        threshold_morph_kernel=args.threshold_morph_kernel,

        color_quantization_k=args.color_quantization_k,
        color_min_component_area_frac=args.color_min_component_area_frac,
        color_morph_kernel=args.color_morph_kernel,

        canny_low=args.canny_low,
        canny_high=args.canny_high,
        edge_morph_kernel=args.edge_morph_kernel,

        min_gradient_support=args.min_gradient_support,
        max_roughness=args.max_roughness,
    )

    for image_path in image_files:
        image_rgb = load_image_rgb(image_path)
        candidates = extract_curve_candidates_from_image(image_rgb, config)

        overlay_rgb = draw_curve_candidates_on_image(
            image_rgb,
            candidates,
            top_k=args.top_k,
            draw_points=False,
        )

        lengths = sorted([len(c.image_points) for c in candidates], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(overlay_rgb)
        axes[1].set_title(f"Improved curves | count={len(candidates)}")
        axes[1].axis("off")

        fig.suptitle(image_path.name)
        plt.tight_layout()
        plt.show()

        print(f"{image_path.name}: {len(candidates)} candidates")
        print(f"Top lengths: {lengths[:10]}")
        for i, cand in enumerate(candidates[:args.top_k]):
            print(
                f"  [{i}] source={cand.source:16s} "
                f"score={cand.score:7.3f} "
                f"closed={cand.closed} "
                f"arc={cand.arc_length:8.2f} "
                f"pts={cand.metadata.get('num_points', -1)} "
                f"grad={cand.metadata.get('gradient_support', -1):.3f} "
                f"rough={cand.metadata.get('roughness', -1):.3f}"
            )


if __name__ == "__main__":
    main()
