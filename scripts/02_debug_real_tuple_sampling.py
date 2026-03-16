from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.io_utils import load_contours_npz
from src.real_patch_adapter import canonicalize_real_contour
from src.real_tuple_generation import build_real_tangent_training_tuple


def plot_patch(ax, patch_points, title, color="tab:blue"):
    ax.plot(patch_points[:, 0], patch_points[:, 1], "-o", color=color, markersize=3)
    center = patch_points[len(patch_points) // 2]
    ax.scatter(center[0], center[1], color="red", s=40)
    ax.set_aspect("equal")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contours_npz", type=str, required=True)
    parser.add_argument("--contour_index", type=int, default=0)
    parser.add_argument("--dense_num_points", type=int, default=300)
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--half_width", type=int, default=12)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--negative_min_offset", type=int, default=5)
    parser.add_argument("--negative_max_offset", type=int, default=25)
    parser.add_argument("--patch_mode", type=str, default="jittered_symmetric")
    parser.add_argument("--jitter_fraction", type=float, default=0.25)
    parser.add_argument("--family", type=str, default="equi_affine")
    args = parser.parse_args()

    contours = load_contours_npz(args.contours_npz)
    contour = contours[args.contour_index]

    canon = canonicalize_real_contour(
        contour,
        dense_num_points=args.dense_num_points,
        target_extent=0.6,
    )

    tup = build_real_tangent_training_tuple(
        contour_xy=contour,
        family=args.family,
        dense_num_points=args.dense_num_points,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        negative_min_offset=args.negative_min_offset,
        negative_max_offset=args.negative_max_offset,
        patch_mode=args.patch_mode,
        jitter_fraction=args.jitter_fraction,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # dense canonical contour
    ax = axes[0, 0]
    ax.plot(canon.canonical_points[:, 0], canon.canonical_points[:, 1], linewidth=1)
    ax.set_title(f"Canonical contour | closed={canon.closed}")
    ax.set_aspect("equal")

    # anchor support on contour
    ax = axes[0, 1]
    ax.plot(canon.canonical_points[:, 0], canon.canonical_points[:, 1], color="lightgray")
    ax.plot(tup.anchor.points[:, 0], tup.anchor.points[:, 1], "-o", color="tab:blue")
    ax.scatter(
        tup.anchor.points[len(tup.anchor.points)//2, 0],
        tup.anchor.points[len(tup.anchor.points)//2, 1],
        color="red",
        s=50,
    )
    ax.set_title("Anchor support")
    ax.set_aspect("equal")

    # positive support
    ax = axes[0, 2]
    transformed_curve = canon.canonical_points @ tup.transform.A.T + tup.transform.b.reshape(1, 2)
    ax.plot(transformed_curve[:, 0], transformed_curve[:, 1], color="lightgray")
    ax.plot(tup.positive.points[:, 0], tup.positive.points[:, 1], "-o", color="tab:green")
    ax.scatter(
        tup.positive.points[len(tup.positive.points)//2, 0],
        tup.positive.points[len(tup.positive.points)//2, 1],
        color="red",
        s=50,
    )
    ax.set_title("Positive support")
    ax.set_aspect("equal")

    plot_patch(axes[1, 0], tup.anchor.centered_points, "Anchor centered patch", color="tab:blue")
    plot_patch(axes[1, 1], tup.positive.centered_points, "Positive centered patch", color="tab:green")

    ax = axes[1, 2]
    for i, neg in enumerate(tup.negatives):
        ax.plot(neg.centered_points[:, 0], neg.centered_points[:, 1], "-o", markersize=2, label=f"neg{i}")
    ax.set_title("Negative centered patches")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
