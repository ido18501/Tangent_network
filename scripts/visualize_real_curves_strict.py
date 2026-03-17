from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.real_curve_generator import (
    RealCurveExtractionConfig,
    extract_curve_candidates_from_image,
)


def _draw_candidates(
    image_bgr: np.ndarray,
    candidates,
    *,
    max_draw: int | None = None,
    label: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    canvas = image_bgr.copy()

    colors = {
        "threshold_region": (0, 255, 0),   # green
        "color_region": (255, 180, 0),     # orange-ish
        "edge": (0, 0, 255),               # red
    }

    draw_list = candidates
    if max_draw is not None:
        draw_list = candidates[:max_draw]

    for i, cand in enumerate(draw_list):
        pts = np.round(cand.image_points).astype(np.int32).reshape(-1, 1, 2)
        color = colors.get(cand.source, (255, 255, 255))

        cv2.polylines(
            canvas,
            [pts],
            isClosed=bool(cand.closed),
            color=color,
            thickness=line_thickness,
        )

        if label and len(cand.image_points) > 0:
            x, y = cand.image_points[0]
            tx = int(np.clip(round(x), 0, canvas.shape[1] - 1))
            ty = int(np.clip(round(y), 0, canvas.shape[0] - 1))
            text = f"{i}: {cand.source} s={cand.score:.2f}"
            cv2.putText(
                canvas,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def _make_edge_debug_image(image_bgr: np.ndarray, blur_ksize: int, blur_sigma: float,
                           canny_low: int, canny_high: int, morph_kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

    edges = cv2.Canny(gray, canny_low, canny_high)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size),
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/strict_curve_debug")
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Much stricter config than before.
    config = RealCurveExtractionConfig(
        min_contour_points=60,
        min_arc_length=180.0,
        min_bbox_diag_frac=0.18,
        max_candidates_per_image=20,
        dedup_center_dist_frac=0.06,
        dedup_length_rel_tol=0.18,

        closed_endpoint_tol=4.0,

        gaussian_blur_ksize=11,
        gaussian_blur_sigma=2.2,
        simplify_epsilon_frac=0.006,
        contour_smooth_window=9,
        contour_smooth_passes=3,

        enable_threshold_regions=True,
        num_threshold_levels=5,
        threshold_min_component_area_frac=0.01,
        threshold_morph_kernel=7,

        enable_color_regions=True,
        color_quantization_k=4,
        color_min_component_area_frac=0.012,
        color_morph_kernel=7,

        enable_edges=True,
        canny_low=60,
        canny_high=140,
        edge_morph_kernel=5,

        min_gradient_support=0.22,
        max_roughness=1.25,

        canonical_dense_num_points=300,
        target_extent=0.6,
    )

    candidates = extract_curve_candidates_from_image(image, config)

    print(f"Found {len(candidates)} candidates")
    for i, cand in enumerate(candidates):
        print(
            f"[{i}] "
            f"source={cand.source:16s} "
            f"score={cand.score:7.3f} "
            f"closed={cand.closed} "
            f"arc={cand.arc_length:8.2f} "
            f"pts={cand.metadata.get('num_points', -1)} "
            f"grad={cand.metadata.get('gradient_support', -1):.3f} "
            f"rough={cand.metadata.get('roughness', -1):.3f} "
            f"fill={cand.metadata.get('fill_ratio', -1):.3f} "
            f"bbox_diag={cand.metadata.get('bbox_diag', -1):.2f}"
        )

    # Save original
    original_out = out_dir / "01_original.png"
    cv2.imwrite(str(original_out), image)

    # Save edge debug
    edge_img = _make_edge_debug_image(
        image,
        blur_ksize=config.gaussian_blur_ksize,
        blur_sigma=config.gaussian_blur_sigma,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
        morph_kernel_size=config.edge_morph_kernel,
    )
    edge_out = out_dir / "02_edges.png"
    cv2.imwrite(str(edge_out), edge_img)

    # Save all candidates
    all_vis = _draw_candidates(
        image,
        candidates,
        max_draw=None,
        label=True,
        line_thickness=2,
    )
    all_out = out_dir / "03_all_candidates.png"
    cv2.imwrite(str(all_out), all_vis)

    # Save top-k
    top_vis = _draw_candidates(
        image,
        candidates,
        max_draw=args.top_k,
        label=True,
        line_thickness=2,
    )
    top_out = out_dir / "04_top_k_candidates.png"
    cv2.imwrite(str(top_out), top_vis)

    # Save best only
    if len(candidates) > 0:
        best_vis = _draw_candidates(
            image,
            [candidates[0]],
            max_draw=1,
            label=True,
            line_thickness=3,
        )
        best_out = out_dir / "05_best_candidate.png"
        cv2.imwrite(str(best_out), best_vis)

    print(f"Saved debug outputs to: {out_dir}")


if __name__ == "__main__":
    main()
