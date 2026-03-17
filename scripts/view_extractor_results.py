from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def list_matching_files(folder: Path, suffix: str) -> list[Path]:
    return sorted(folder.glob(f"*{suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/outputs/improved_extractor_debug",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "curves_only", "single_curves"],
        default="debug",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if args.mode == "debug":
        files = list_matching_files(results_dir, "_debug.png")
    elif args.mode == "curves_only":
        files = list_matching_files(results_dir, "_curves_only.png")
    else:
        files = sorted((results_dir / "single_curves").glob("*.png"))

    files = files[: args.max_images]

    if not files:
        raise RuntimeError(f"No files found for mode={args.mode} in {results_dir}")

    for path in files:
        img = mpimg.imread(path)

        plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.title(path.name)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
