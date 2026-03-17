from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.real_curve_generator import (
    RealCurveExtractionConfig,
    draw_curve_candidates_on_image,
    extract_curve_candidates_from_image,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="debug_real_curves.png", help="Output visualization path")
    parser.add_argument("--top_k", type=int, default=12, help="How many top candidates to draw")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    config = RealCurveExtractionConfig()
    candidates = extract_curve_candidates_from_image(image, config)

    print(f"Found {len(candidates)} candidates")
    for i, cand in enumerate(candidates[:args.top_k]):
        print(
            f"[{i}] source={cand.source:16s} "
            f"score={cand.score:7.3f} "
            f"closed={cand.closed} "
            f"arc={cand.arc_length:8.2f} "
            f"pts={cand.metadata.get('num_points', -1)} "
            f"grad={cand.metadata.get('gradient_support', -1):.3f} "
            f"rough={cand.metadata.get('roughness', -1):.3f}"
        )

    vis = draw_curve_candidates_on_image(
        image_bgr=image,
        candidates=candidates,
        top_k=args.top_k,
        draw_points=False,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), vis)
    if not ok:
        raise RuntimeError(f"Failed writing output image to {out_path}")

    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
