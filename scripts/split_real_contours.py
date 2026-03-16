from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np


def main() -> None:
    src_dir = Path("data/outputs/extract_contours/contours_npz")

    out_root = Path("data/outputs/extract_contours")
    train_dir = out_root / "train_contours_npz"
    val_dir = out_root / "val_contours_npz"
    test_dir = out_root / "test_contours_npz"

    seed = 123
    train_frac = 0.70
    val_frac = 0.15
    test_frac = 0.15

    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-12:
        raise ValueError("Fractions must sum to 1.")

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    files = sorted(src_dir.glob("*_contours.npz"))
    if not files:
        raise RuntimeError(f"No contour files found in {src_dir}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    files = [files[i] for i in perm]

    n = len(files)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    if len(test_files) != n_test:
        raise RuntimeError("Split size mismatch.")

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy instead of move, to preserve the original folder.
    for fp in train_files:
        shutil.copy2(fp, train_dir / fp.name)

    for fp in val_files:
        shutil.copy2(fp, val_dir / fp.name)

    for fp in test_files:
        shutil.copy2(fp, test_dir / fp.name)

    print("Done.")
    print(f"source files: {n}")
    print(f"train files:  {len(train_files)} -> {train_dir}")
    print(f"val files:    {len(val_files)} -> {val_dir}")
    print(f"test files:   {len(test_files)} -> {test_dir}")
    print("train names:")
    for fp in train_files:
        print(" ", fp.name)
    print("val names:")
    for fp in val_files:
        print(" ", fp.name)
    print("test names:")
    for fp in test_files:
        print(" ", fp.name)


if __name__ == "__main__":
    main()
