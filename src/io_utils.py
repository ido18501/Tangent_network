from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_image_files(folder: str | Path):
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def save_contours_npz(output_path: str | Path, contours_xy: List[np.ndarray]) -> None:
    output_path = Path(output_path)
    payload = {f"contour_{i}": c.astype(np.float64) for i, c in enumerate(contours_xy)}
    payload["num_contours"] = np.array([len(contours_xy)], dtype=np.int64)
    np.savez_compressed(output_path, **payload)


def load_contours_npz(path: str | Path) -> list[np.ndarray]:
    data = np.load(path, allow_pickle=False)
    num_contours = int(data["num_contours"][0])
    contours = [data[f"contour_{i}"].astype(np.float64) for i in range(num_contours)]
    return contours
