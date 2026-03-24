
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    generate_random_piecewise_curve,
    fit_curve_to_canvas_with_random_size,
    warp_curve_sampling,
)
from utils.derivatives import compute_euclidean_arc_length_derivatives


def sample_curve(
    rng: np.random.Generator,
    *,
    num_curve_points: int = 300,
    curve_family_probs: dict[str, float] | None = None,
    fourier_max_freq: int = 5,
    fourier_scale: float = 0.9,
    fourier_decay_power: float = 2.0,
    curve_max_tries: int = 300,
    curve_min_size: float = 0.45,
    curve_max_size: float = 0.75,
    closed: bool = True,
    warp_sampling_prob: float = 0.7,
    warp_sampling_strength: float = 0.18,
) -> np.ndarray:
    if curve_family_probs is None:
        curve_family_probs = {"fourier": 1.0, "piecewise": 0.0}

    names = list(curve_family_probs.keys())
    probs = np.asarray([curve_family_probs[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    family = str(rng.choice(names, p=probs))

    if family == "fourier":
        t = np.linspace(0.0, 2.0 * np.pi, num_curve_points, endpoint=False)
        curve_points, _ = generate_random_simple_fourier_curve(
            t=t,
            max_freq=fourier_max_freq,
            scale=fourier_scale,
            decay_power=fourier_decay_power,
            rng=rng,
            max_tries=curve_max_tries,
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
        raise ValueError(f"Unsupported sampled curve family: {family}")

    if rng.random() < warp_sampling_prob:
        curve_points = warp_curve_sampling(
            curve_points,
            rng=rng,
            strength=warp_sampling_strength,
            closed=closed,
        )
    return np.asarray(curve_points, dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def collect_examples(
    *,
    num_curves: int,
    anchors_per_curve: int,
    dense_num_points: int,
    seed: int,
    num_curve_points: int,
    curve_family_probs: dict[str, float],
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for curve_id in range(num_curves):
        curve = sample_curve(
            rng,
            num_curve_points=num_curve_points,
            curve_family_probs=curve_family_probs,
        )
        n = len(curve)
        anchor_indices = rng.choice(n, size=min(anchors_per_curve, n), replace=False)

        for anchor_idx in anchor_indices:
            gt_first, gt_second, dense_anchor = compute_euclidean_arc_length_derivatives(
                curve, int(anchor_idx), dense_num_points=dense_num_points
            )
            t = normalize(gt_first)
            s = normalize(gt_second)
            second_norm = float(np.linalg.norm(gt_second))
            cos_ts = float(np.clip(np.dot(t, s), -1.0, 1.0)) if second_norm > 1e-12 else np.nan
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_ts, -1.0, 1.0)))) if second_norm > 1e-12 else np.nan
            angle_dev_from_90 = float(abs(angle_deg - 90.0)) if second_norm > 1e-12 else np.nan

            rows.append(
                {
                    "curve_id": curve_id,
                    "anchor_index": int(anchor_idx),
                    "curve": curve,
                    "anchor_point": dense_anchor,
                    "gt_first": gt_first,
                    "gt_second": gt_second,
                    "gt_first_unit": t,
                    "gt_second_unit": s,
                    "gt_second_norm": second_norm,
                    "cos_tangent_second": cos_ts,
                    "angle_deg": angle_deg,
                    "angle_dev_from_90_deg": angle_dev_from_90,
                }
            )
    return rows


def draw_example(ax, row: dict, title: str, vector_scale: float = 0.08) -> None:
    curve = row["curve"]
    p = row["anchor_point"]
    t = row["gt_first_unit"]
    s = row["gt_second_unit"]
    k = row["gt_second_norm"]

    ax.plot(curve[:, 0], curve[:, 1], linewidth=1.0)
    ax.scatter([p[0]], [p[1]], s=25)

    # tangent
    ax.arrow(
        p[0], p[1],
        vector_scale * t[0], vector_scale * t[1],
        length_includes_head=True, head_width=0.015
    )
    # second derivative direction
    if np.isfinite(k) and k > 1e-12:
        ax.arrow(
            p[0], p[1],
            vector_scale * s[0], vector_scale * s[1],
            length_includes_head=True, head_width=0.015
        )

    ax.set_title(
        f"{title}\n|x''|={k:.4f}, angle_dev90={row['angle_dev_from_90_deg']:.2f}°",
        fontsize=9
    )
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-curves", type=int, default=40)
    p.add_argument("--anchors-per-curve", type=int, default=50)
    p.add_argument("--dense-num-points", type=int, default=4096)
    p.add_argument("--num-curve-points", type=int, default=300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--family-probs", type=str, default="fourier:1.0,piecewise:0.0")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_family_probs = {}
    for part in args.family_probs.split(","):
        name, value = part.split(":")
        curve_family_probs[name.strip()] = float(value)

    rows = collect_examples(
        num_curves=args.num_curves,
        anchors_per_curve=args.anchors_per_curve,
        dense_num_points=args.dense_num_points,
        seed=args.seed,
        num_curve_points=args.num_curve_points,
        curve_family_probs=curve_family_probs,
    )

    valid_rows = [r for r in rows if np.isfinite(r["gt_second_norm"]) and r["gt_second_norm"] > 1e-12]
    valid_rows.sort(key=lambda r: r["gt_second_norm"])

    norms = np.array([r["gt_second_norm"] for r in valid_rows], dtype=np.float64)
    angle_dev = np.array([r["angle_dev_from_90_deg"] for r in valid_rows], dtype=np.float64)
    abs_cos = np.abs(np.array([r["cos_tangent_second"] for r in valid_rows], dtype=np.float64))

    # summary
    summary = {
        "num_examples_total": len(rows),
        "num_examples_valid_second": len(valid_rows),
        "gt_second_norm_quantiles": {
            "q00": float(np.quantile(norms, 0.00)),
            "q10": float(np.quantile(norms, 0.10)),
            "q25": float(np.quantile(norms, 0.25)),
            "q50": float(np.quantile(norms, 0.50)),
            "q75": float(np.quantile(norms, 0.75)),
            "q90": float(np.quantile(norms, 0.90)),
            "q100": float(np.quantile(norms, 1.00)),
        },
        "angle_dev_from_90_deg_quantiles": {
            "q10": float(np.quantile(angle_dev, 0.10)),
            "q25": float(np.quantile(angle_dev, 0.25)),
            "q50": float(np.quantile(angle_dev, 0.50)),
            "q75": float(np.quantile(angle_dev, 0.75)),
            "q90": float(np.quantile(angle_dev, 0.90)),
        },
        "abs_cos_tangent_second_quantiles": {
            "q10": float(np.quantile(abs_cos, 0.10)),
            "q25": float(np.quantile(abs_cos, 0.25)),
            "q50": float(np.quantile(abs_cos, 0.50)),
            "q75": float(np.quantile(abs_cos, 0.75)),
            "q90": float(np.quantile(abs_cos, 0.90)),
        },
    }

    with open(out_dir / "gt_second_direction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Scatter plots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(norms, angle_dev, s=8, alpha=0.5)
    ax1.set_xlabel("||GT second derivative||")
    ax1.set_ylabel("Angle deviation from 90° (deg)")
    ax1.set_title("Perpendicularity error vs curvature magnitude")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(norms, abs_cos, s=8, alpha=0.5)
    ax2.set_xlabel("||GT second derivative||")
    ax2.set_ylabel("|cos(tangent, second)|")
    ax2.set_title("Absolute tangent leakage vs curvature magnitude")

    fig.tight_layout()
    fig.savefig(out_dir / "gt_second_perpendicularity_vs_curvature.png", dpi=180)
    plt.close(fig)

    # Binned averages
    num_bins = min(12, max(4, len(norms) // 150))
    bin_edges = np.quantile(norms, np.linspace(0.0, 1.0, num_bins + 1))
    bin_centers = []
    bin_mean_dev = []
    bin_mean_abscos = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (norms >= lo) & (norms <= hi if i == num_bins - 1 else norms < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(float(norms[mask].mean()))
        bin_mean_dev.append(float(angle_dev[mask].mean()))
        bin_mean_abscos.append(float(abs_cos[mask].mean()))

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(bin_centers, bin_mean_dev, marker="o")
    ax1.set_xlabel("Mean ||GT second derivative|| per bin")
    ax1.set_ylabel("Mean angle deviation from 90° (deg)")
    ax1.set_title("Binned perpendicularity error vs curvature magnitude")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(bin_centers, bin_mean_abscos, marker="o")
    ax2.set_xlabel("Mean ||GT second derivative|| per bin")
    ax2.set_ylabel("Mean |cos(tangent, second)|")
    ax2.set_title("Binned tangent leakage vs curvature magnitude")

    fig.tight_layout()
    fig.savefig(out_dir / "gt_second_perpendicularity_binned.png", dpi=180)
    plt.close(fig)

    # Visual examples across increasing curvature
    n = len(valid_rows)
    sample_positions = np.linspace(0, n - 1, 12, dtype=int)
    chosen = [valid_rows[i] for i in sample_positions]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, row, pos in zip(axes.flat, chosen, sample_positions):
        title = f"rank {pos+1}/{n}"
        draw_example(ax, row, title)
    fig.tight_layout()
    fig.savefig(out_dir / "gt_examples_increasing_curvature.png", dpi=180)
    plt.close(fig)

    # Low-curvature closeups
    low_count = min(12, len(valid_rows))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, row in zip(axes.flat, valid_rows[:low_count]):
        draw_example(ax, row, "lowest-curvature")
    fig.tight_layout()
    fig.savefig(out_dir / "gt_examples_low_curvature.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
