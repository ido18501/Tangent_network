import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    generate_random_piecewise_curve,
    fit_curve_to_canvas_with_random_size,
    warp_curve_sampling,
)


def add_curve_noise(
    curve_points: np.ndarray,
    rng: np.random.Generator,
    point_noise_std: float = 0.003,
    orthogonal_noise_std: float = 0.006,
) -> np.ndarray:
    pts = np.asarray(curve_points, dtype=np.float64).copy()

    if point_noise_std > 0.0:
        pts += rng.normal(0.0, point_noise_std, size=pts.shape)

    if orthogonal_noise_std > 0.0:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        tang = next_pts - prev_pts
        tang_norm = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = tang / np.clip(tang_norm, 1e-12, None)
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        coeff = rng.normal(0.0, orthogonal_noise_std, size=(len(pts), 1))
        pts = pts + coeff * normal

    return pts


def sample_curve_family(rng: np.random.Generator, probs: dict[str, float]) -> str:
    names = list(probs.keys())
    p = np.asarray([probs[n] for n in names], dtype=np.float64)
    p = p / p.sum()
    return str(rng.choice(names, p=p))


def generate_one_curve(
    rng: np.random.Generator,
    num_curve_points: int = 400,
    curve_family_probs: dict[str, float] | None = None,
    closed: bool = True,
    fourier_max_freq: int = 9,
    fourier_scale: float = 1.1,
    fourier_decay_power: float = 1.35,
    curve_min_size: float = 0.25,
    curve_max_size: float = 0.95,
    warp_sampling_prob: float = 0.7,
    warp_sampling_strength: float = 0.18,
    point_noise_std: float = 0.003,
    orthogonal_noise_std: float = 0.006,
) -> tuple[np.ndarray, str]:
    if curve_family_probs is None:
        curve_family_probs = {"fourier": 0.55, "piecewise": 0.45}

    family = sample_curve_family(rng, curve_family_probs)

    if family == "fourier":
        t = np.linspace(0.0, 2.0 * np.pi, num_curve_points, endpoint=False)
        curve_points, _ = generate_random_simple_fourier_curve(
            t=t,
            max_freq=fourier_max_freq,
            scale=fourier_scale,
            decay_power=fourier_decay_power,
            rng=rng,
            max_tries=300,
            center=True,
            fit_to_canvas=True,
            min_size=curve_min_size,
            max_size=curve_max_size,
        )
    elif family == "piecewise":
        curve_points = generate_random_piecewise_curve(
            num_points=num_curve_points,
            rng=rng,
            closed=closed,
        )
        curve_points = fit_curve_to_canvas_with_random_size(
            curve_points,
            rng=rng,
            min_size=curve_min_size,
            max_size=curve_max_size,
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    if rng.random() < warp_sampling_prob:
        curve_points = warp_curve_sampling(
            curve_points,
            rng=rng,
            strength=warp_sampling_strength,
            closed=closed,
        )

    curve_points = add_curve_noise(
        curve_points,
        rng=rng,
        point_noise_std=point_noise_std,
        orthogonal_noise_std=orthogonal_noise_std,
    )

    return curve_points, family


def main():
    rng = np.random.default_rng(0)

    curves = []
    families = []

    for _ in range(50):
        curve, family = generate_one_curve(rng)
        curves.append(curve)
        families.append(family)

    fig, axes = plt.subplots(5, 10, figsize=(20, 10))

    for ax, curve, family in zip(axes.flat, curves, families):
        ax.plot(curve[:, 0], curve[:, 1], linewidth=1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(family, fontsize=8)

    plt.suptitle("Sample of 50 realistic synthetic training curves", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
