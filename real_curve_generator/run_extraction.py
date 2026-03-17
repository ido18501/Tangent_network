from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from real_curve_generator.dataset_adapter import process_image_to_samples, read_image, save_samples_npz
else:
    from .dataset_adapter import process_image_to_samples, read_image, save_samples_npz


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract robust real-image curves for Tangent Network.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dense_points", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(input_dir):
        image = read_image(image_path)
        image_id = image_path.stem
        samples = process_image_to_samples(
            image,
            image_id=image_id,
            dense_points=args.dense_points,
            top_k=args.top_k,
        )
        out_path = output_dir / f"{image_id}.npz"
        save_samples_npz(samples, out_path)
        print(f"{image_path.name}: saved {len(samples)} curves -> {out_path}")


if __name__ == "__main__":
    main()
