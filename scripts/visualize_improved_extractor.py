from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.io_utils import ensure_dir, list_image_files

# CHANGE THIS import to your new extractor module / function
# Example:
# from src.contour_extraction_v2 import extract_curves_from_image, ExtractorConfig
from src.contour_extraction import extract_contours, ContourExtractionConfig, load_image


def save_curves_npz(path: Path, curves: list[np.ndarray]) -> None:
    payload = {f"curve_{i}": c.astype(np.float64) for i, c in enumerate(curves)}
    payload["num_curves"] = np.array([len(curves)], dtype=np.int64)
    np.savez_compressed(path, **payload)


def draw_curves_on_image(
    image_bgr: np.ndarray,
    curves: list[np.ndarray],
    max_curves: int | None = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    use_curves = curves if max_curves is None else curves[:max_curves]

    for curve in use_curves:
        if len(curve) < 2:
            continue
        pts = np.round(curve).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

    return canvas


def save_curves_only_figure(curves: list[np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    for curve in curves:
        if len(curve) < 2:
            continue
        ax.plot(curve[:, 0], curve[:, 1], linewidth=1)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_single_curve_debug(curve: np.ndarray, output_path: Path, title: str, step: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(curve[:, 0], curve[:, 1], "-o", markersize=2)
    for i in range(0, len(curve), step):
        x, y = curve[i]
        ax.text(x, y, str(i), fontsize=6)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_debug_figure(
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    curves: list[np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    lengths = [len(c) for c in curves]
    lengths_sorted = sorted(lengths, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Curves overlay | count={len(curves)}")
    axes[1].axis("off")

    axes[2].bar(np.arange(min(len(lengths_sorted), 20)), lengths_sorted[:20])
    axes[2].set_title("Top curve lengths")
    axes[2].set_xlabel("curve rank")
    axes[2].set_ylabel("num points")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images" / "BSR" / "BSDS500" / "data" / "images" / "train"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "outputs" / "improved_extractor_debug"),
    )
    parser.add_argument("--max_images", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "npz")
    ensure_dir(output_dir / "single_curves")

    image_files = list_image_files(input_dir)
    if args.max_images is not None:
        image_files = image_files[:args.max_images]

    if not image_files:
        raise RuntimeError(f"No image files found in {input_dir}")

    # CHANGE THIS config block to your improved extractor config
    config = ContourExtractionConfig(
        blur_kernel_size=5,
        canny_low=80,
        canny_high=160,
        min_contour_points=30,
        min_arc_length=40.0,
        polygon_epsilon_frac=0.0,
    )

    print(f"Found {len(image_files)} image(s)")
    print("Running improved extractor visualization...")

    for image_path in image_files:
        image_bgr = load_image(str(image_path))

        # CHANGE THIS call to your improved extractor call
        gray, edges, curves = extract_contours(image_bgr, config)

        curves = sorted(curves, key=len, reverse=True)

        overlay_bgr = draw_curves_on_image(image_bgr, curves, max_curves=50)

        save_debug_figure(
            image_bgr=image_bgr,
            overlay_bgr=overlay_bgr,
            curves=curves,
            output_path=output_dir / f"{image_path.stem}_debug.png",
            title=image_path.name,
        )

        save_curves_only_figure(
            curves=curves[:50],
            output_path=output_dir / f"{image_path.stem}_curves_only.png",
            title=f"{image_path.name} curves",
        )

        save_curves_npz(output_dir / "npz" / f"{image_path.stem}_curves.npz", curves)

        for i, curve in enumerate(curves[:3]):
            save_single_curve_debug(
                curve=curve,
                output_path=output_dir / "single_curves" / f"{image_path.stem}_curve_{i:02d}.png",
                title=f"{image_path.name} | curve {i} | len={len(curve)}",
                step=max(1, len(curve) // 20),
            )

        print(f"[OK] {image_path.name}: {len(curves)} curves")

    print("Done.")


if __name__ == "__main__":
    main()
