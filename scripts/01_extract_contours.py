from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import argparse
import sys
from pathlib import Path
from io_utils import ensure_dir, list_image_files, save_contours_npz
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from contour_extraction import (  # noqa: E402
    ContourExtractionConfig,
    draw_contours_on_image,
    extract_contours,
    load_image,
)
from io_utils import ensure_dir, list_image_files  # noqa: E402


def save_debug_figure(
    image_bgr,
    gray,
    edges,
    overlay_bgr,
    output_path: Path,
    title: str,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Gray")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Edges")
    axes[2].axis("off")

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Contours overlay")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "images" / "raw"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "outputs" / "extract_contours"),
    )
    parser.add_argument("--blur_kernel_size", type=int, default=5)
    parser.add_argument("--canny_low", type=int, default=80)
    parser.add_argument("--canny_high", type=int, default=160)
    parser.add_argument("--min_contour_points", type=int, default=30)
    parser.add_argument("--min_arc_length", type=float, default=40.0)
    parser.add_argument("--polygon_epsilon_frac", type=float, default=0.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    contours_output_dir = output_dir / "contours_npz"
    ensure_dir(contours_output_dir)

    image_files = list_image_files(input_dir)[:8]
    if not image_files:
        raise RuntimeError(f"No image files found in {input_dir}")

    config = ContourExtractionConfig(
        blur_kernel_size=args.blur_kernel_size,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_contour_points=args.min_contour_points,
        min_arc_length=args.min_arc_length,
        polygon_epsilon_frac=args.polygon_epsilon_frac,
    )

    print(f"Found {len(image_files)} image(s) in {input_dir}")
    print("Running contour extraction...")

    for image_path in image_files:
        image_bgr = load_image(str(image_path))
        gray, edges, contours_xy = extract_contours(image_bgr, config)
        overlay_bgr = draw_contours_on_image(image_bgr, contours_xy, draw_points=False)

        out_name = image_path.stem + "_debug.png"
        out_path = output_dir / out_name

        save_debug_figure(
            image_bgr=image_bgr,
            gray=gray,
            edges=edges,
            overlay_bgr=overlay_bgr,
            output_path=out_path,
            title=f"{image_path.name} | contours={len(contours_xy)}",
        )

        contours_npz_path = contours_output_dir / f"{image_path.stem}_contours.npz"
        save_contours_npz(contours_npz_path, contours_xy)

        print(
            f"[OK] {image_path.name}: extracted {len(contours_xy)} contours "
            f"-> {out_path} | {contours_npz_path}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
