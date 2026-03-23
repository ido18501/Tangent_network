from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.curve_generation import generate_random_simple_fourier_curve
from utils.derivatives import compute_euclidean_arc_length_derivatives


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-curves", type=int, default=4)
    p.add_argument("--num-points", type=int, default=300)
    p.add_argument("--max-freq", type=int, default=5)
    p.add_argument("--scale", type=float, default=0.9)
    p.add_argument("--decay-power", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", type=str, default="euclidean_derivatives_demo.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(2, max(2, args.num_curves // 2), figsize=(12, 7))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes[:args.num_curves]:
        t = np.linspace(0.0, 2.0 * np.pi, args.num_points, endpoint=False)
        curve, _ = generate_random_simple_fourier_curve(
            t=t,
            max_freq=args.max_freq,
            scale=args.scale,
            decay_power=args.decay_power,
            rng=rng,
            max_tries=200,
            center=True,
            fit_to_canvas=True,
            min_size=0.45,
            max_size=0.75,
        )
        idx = int(rng.integers(0, len(curve)))
        d1, d2, _ = compute_euclidean_arc_length_derivatives(curve, idx, dense_num_points=4096)
        p = curve[idx]

        ax.plot(curve[:, 0], curve[:, 1], linewidth=1.2)
        ax.scatter([p[0]], [p[1]], s=18)
        ax.arrow(p[0], p[1], 0.08 * d1[0], 0.08 * d1[1], head_width=0.02, length_includes_head=True)
        ax.arrow(p[0], p[1], 0.015 * d2[0], 0.015 * d2[1], head_width=0.02, length_includes_head=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"idx={idx}")

    for ax in axes[args.num_curves:]:
        ax.axis("off")

    fig.suptitle("Euclidean numerical first/second derivatives on Fourier curves")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
