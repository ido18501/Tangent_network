from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from subprocess import run

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_items(items: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:n_train + n_val + n_test],
    }


def copy_split(split_map: dict[str, list[Path]], out_root: Path) -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {}
    for split, files in split_map.items():
        split_dir = out_root / split
        ensure_dir(split_dir)
        manifest[split] = []

        for src in files:
            dst = split_dir / src.name

            # avoid silent overwrites if duplicate basenames exist
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                parent_tag = src.parent.name
                dst = split_dir / f"{stem}__{parent_tag}{suffix}"

            shutil.copy2(src, dst)
            manifest[split].append(str(dst))
    return manifest


from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import Popen


def chunk_list(items, n_chunks):
    n = max(1, n_chunks)
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks


def copy_chunk(files: list[Path], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for src in files:
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)


def run_one_extraction(input_dir: Path, output_dir: Path) -> int:
    ensure_dir(output_dir)
    cmd = [
        "python",
        "real_curve_generator/run_extraction.py",
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd))
    proc = Popen(cmd)
    return proc.wait()


def merge_npz_files(chunk_output_dirs: list[Path], final_output_dir: Path) -> None:
    ensure_dir(final_output_dir)
    for chunk_dir in chunk_output_dirs:
        for npz_file in chunk_dir.glob("*.npz"):
            dst = final_output_dir / npz_file.name
            if dst.exists():
                raise RuntimeError(f"Duplicate output file during merge: {dst}")
            shutil.move(str(npz_file), str(dst))


def run_curve_extraction_parallel(split_image_root: Path, split_curve_root: Path, num_workers: int) -> None:
    for split in ["train", "val", "test"]:
        in_dir = split_image_root / split
        out_dir = split_curve_root / split
        ensure_dir(out_dir)

        images = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if not images:
            print(f"No images found in {in_dir}, skipping.")
            continue

        workers = min(num_workers, len(images))
        chunks = chunk_list(images, workers)

        tmp_root = split_curve_root / f"_{split}_chunks"
        ensure_dir(tmp_root)

        chunk_input_dirs = []
        chunk_output_dirs = []

        print(f"Preparing {split}: {len(images)} images across {workers} workers")

        for w, chunk in enumerate(chunks):
            chunk_in = tmp_root / f"{split}_input_{w:02d}"
            chunk_out = tmp_root / f"{split}_output_{w:02d}"
            copy_chunk(chunk, chunk_in)
            ensure_dir(chunk_out)
            chunk_input_dirs.append(chunk_in)
            chunk_output_dirs.append(chunk_out)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(run_one_extraction, cin, cout): (cin, cout)
                for cin, cout in zip(chunk_input_dirs, chunk_output_dirs)
            }
            for fut in as_completed(futures):
                cin, cout = futures[fut]
                rc = fut.result()
                if rc != 0:
                    raise RuntimeError(f"Extraction failed for chunk {cin} -> {cout} with code {rc}")

        merge_npz_files(chunk_output_dirs, out_dir)

        shutil.rmtree(tmp_root)
        print(f"Finished {split}: outputs merged into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw_real_images")
    parser.add_argument("--split_image_dir", type=str, default="data/real_images_split")
    parser.add_argument("--split_curve_dir", type=str, default="data/real_curves_split")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/real_image_split.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    split_image_dir = Path(args.split_image_dir)
    split_curve_dir = Path(args.split_curve_dir)
    manifest_path = Path(args.manifest_path)

    ensure_dir(split_image_dir)
    ensure_dir(split_curve_dir)
    ensure_dir(manifest_path.parent)

    images = list_images(raw_dir)
    if args.max_images is not None and len(images) > args.max_images:
        rng = random.Random(args.seed)
        images = images.copy()
        rng.shuffle(images)
        images = sorted(images[:args.max_images])
    if not images:
        raise RuntimeError(f"No images found in {raw_dir}")

    print(f"Found {len(images)} images in {raw_dir}")

    split_map = split_items(images, args.train_ratio, args.val_ratio, args.seed)

    print(
        "Split sizes:",
        {k: len(v) for k, v in split_map.items()}
    )

    manifest = copy_split(split_map, split_image_dir)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_dir": str(raw_dir),
                "split_image_dir": str(split_image_dir),
                "split_curve_dir": str(split_curve_dir),
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "counts": {k: len(v) for k, v in split_map.items()},
                "files": manifest,
            },
            f,
            indent=2,
        )

    print(f"Saved manifest to {manifest_path}")

    run_curve_extraction_parallel(split_image_dir, split_curve_dir, args.num_workers)

    print("Done.")


if __name__ == "__main__":
    main()
