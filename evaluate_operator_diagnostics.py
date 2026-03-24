from __future__ import annotations
import os
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running from project_root/scripts or project_root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT_CANDIDATES = [THIS_FILE.parent, THIS_FILE.parent.parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from utils.patch_sampling import sample_patch_around_index
from utils.transformations import apply_transformation, sample_transformation
from utils.derivatives import compute_euclidean_arc_length_derivatives


@dataclass
class SampleDiagnostics:
    index: int
    family: str
    anchor_center_index: int
    patch: np.ndarray
    positive_patch: np.ndarray
    full_curve_centered: np.ndarray
    full_curve_transformed_centered: np.ndarray
    gt_first: np.ndarray
    gt_second: np.ndarray
    pred_first: np.ndarray
    pred_second: np.ndarray
    gt_first_pos: np.ndarray
    gt_second_pos: np.ndarray
    pred_first_pos: np.ndarray
    pred_second_pos: np.ndarray
    transform_matrix: np.ndarray
    first_cos: float
    first_angle_deg: float
    first_rmse: float
    second_cos: float
    second_angle_deg: float
    second_rmse: float
    second_gt_norm: float
    pred_first_norm: float
    pred_second_norm: float
    mean_step: float
    median_step: float
    patch_arc: float
    patch_chord: float
    spacing_std: float
    spacing_cv: float
    gap_ratio: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tangent-operator model on the test set and produce diagnostics.")
    p.add_argument("--run-dir", type=str, required=True, help="Training run directory containing checkpoints/best_model.pt")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override. Defaults to <run-dir>/checkpoints/best_model.pt")
    p.add_argument("--output-dir", type=str, default=None, help="Optional output directory. Defaults to <run-dir>/diagnostics")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test-curve-dir", type=str, default=None, help="Optional override for test curve dir")
    p.add_argument("--test-length", type=int, default=None, help="Optional override for number of test samples")
    p.add_argument("--num-visualizations", type=int, default=6)
    p.add_argument("--high-curvature-threshold", type=float, default=1.0)
    p.add_argument("--seed-offset", type=int, default=20_000, help="Must match training script's test seed offset")
    return p.parse_args()


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_int_list(text: str | list[int]) -> list[int]:
    if isinstance(text, list):
        return [int(x) for x in text]
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_family_probs(text: str | dict[str, float]) -> dict[str, float]:
    if isinstance(text, dict):
        return {str(k): float(v) for k, v in text.items()}
    out: dict[str, float] = {}
    for item in str(text).split(","):
        name, value = item.split(":")
        out[name.strip()] = float(value.strip())
    return out


def cosine_and_angle(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    pred_n = pred / max(np.linalg.norm(pred), eps)
    tgt_n = target / max(np.linalg.norm(target), eps)
    cos = float(np.clip(np.dot(pred_n, tgt_n), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    return cos, ang


def make_test_dataset(args: argparse.Namespace) -> TangentDataset:
    test_curve_dir = args.test_curve_dir or "data/precomputed_curves_fourier_1000/test"
    test_length = args.test_length if args.test_length is not None else 500

    return TangentDataset(
        length=test_length,
        family="euclidean",
        num_curve_points=1000,
        patch_size=9,
        half_width=12,
        num_negatives=0,
        negative_min_offset=5,
        negative_max_offset=25,
        negative_other_curve_fraction=0.5,
        patch_mode="random_warp_symmetric",
        jitter_fraction=0.25,
        closed=True,
        return_centered=True,
        point_noise_std=0.0,
        curve_family_probs={"fourier": 1.0, "piecewise": 0.0},
        warp_sampling_prob=0.7,
        warp_sampling_strength=0.18,
        orthogonal_noise_std=0.0,
        gt_dense_num_points=4096,
        seed=123 + int(args.seed_offset),
        use_precomputed_curves=True,
        precomputed_curve_dir=test_curve_dir,
    )


def build_model(device: torch.device) -> TangentOperatorModel:
    model = TangentOperatorModel(
        patch_size=9,
        point_mlp_dims=[64, 64, 128],
        head_dims=[128, 64],
        use_batchnorm=True,
        point_dropout=0.0,
        head_dropout=0.0,
        operator_kernel_size=5,
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def _select_anchor_index(num_points: int, closed: bool, half_width: int, rng: np.random.Generator) -> int:
    if closed:
        return int(rng.integers(0, num_points))
    left = half_width
    right = num_points - half_width
    if left >= right:
        raise ValueError("No valid center indices remain for the requested margin.")
    return int(rng.integers(left, right))


def _load_or_generate_curve(dataset: TangentDataset, index: int, rng: np.random.Generator) -> np.ndarray:
    if dataset.use_precomputed_curves:
        return dataset._load_curve_from_disk(index)
    return dataset._generate_curve(rng)


def reproduce_sample(dataset: TangentDataset, index: int) -> dict[str, Any]:
    rng = dataset._make_rng(index)
    curve = _load_or_generate_curve(dataset, index, rng)
    half_width = dataset._sample_half_width(rng)
    anchor_center_index = _select_anchor_index(len(curve), dataset.closed, half_width, rng)

    anchor_patch = sample_patch_around_index(
        curve_points=curve,
        center_index=anchor_center_index,
        patch_size=dataset.patch_size,
        half_width=half_width,
        mode=dataset.patch_mode,
        closed=dataset.closed,
        rng=rng,
        jitter_fraction=dataset.jitter_fraction,
    )

    transform = sample_transformation(
        family=dataset.family,
        rng=rng,
        **dataset.transform_kwargs,
    )
    positive_patch = apply_transformation(anchor_patch, transform)

    gt_first, gt_second, _ = compute_euclidean_arc_length_derivatives(
        curve_points=curve,
        anchor_index=anchor_center_index,
        dense_num_points=dataset.gt_dense_num_points,
    )

    anchor_center = anchor_patch[dataset.patch_size // 2].copy()
    positive_center = positive_patch[dataset.patch_size // 2].copy()
    curve_centered = curve - anchor_center[None, :]
    curve_transformed = apply_transformation(curve, transform)
    curve_transformed_centered = curve_transformed - positive_center[None, :]

    if dataset.return_centered:
        anchor_patch = anchor_patch - anchor_center[None, :]
        positive_patch = positive_patch - positive_center[None, :]

    return {
        "curve_centered": curve_centered.astype(np.float32),
        "curve_transformed_centered": curve_transformed_centered.astype(np.float32),
        "anchor_patch": anchor_patch.astype(np.float32),
        "positive_patch": positive_patch.astype(np.float32),
        "transform_matrix": np.asarray(transform.A, dtype=np.float32),
        "anchor_center_index": int(anchor_center_index),
        "gt_first": gt_first.astype(np.float32),
        "gt_second": gt_second.astype(np.float32),
        "family": dataset.family,
    }


def predict_patch(model: torch.nn.Module, patch: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(patch).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    pred_first = out["vector_first"][0].detach().cpu().numpy().astype(np.float32)
    pred_second = out["vector_second"][0].detach().cpu().numpy().astype(np.float32)
    return pred_first, pred_second


def patch_spacing_metrics(patch: np.ndarray) -> dict[str, float]:
    d = np.linalg.norm(np.diff(patch, axis=0), axis=1)
    mean_step = float(d.mean())
    median_step = float(np.median(d))
    patch_arc = float(d.sum())
    patch_chord = float(np.linalg.norm(patch[-1] - patch[0]))
    spacing_std = float(d.std())
    spacing_cv = float(spacing_std / (mean_step + 1e-12))
    gap_ratio = float(d.max() / max(d.min(), 1e-12))
    return {
        "mean_step": mean_step,
        "median_step": median_step,
        "patch_arc": patch_arc,
        "patch_chord": patch_chord,
        "spacing_std": spacing_std,
        "spacing_cv": spacing_cv,
        "gap_ratio": gap_ratio,
    }


def evaluate_dataset(model: torch.nn.Module, dataset: TangentDataset, device: torch.device) -> list[SampleDiagnostics]:
    rows: list[SampleDiagnostics] = []
    for idx in range(len(dataset)):
        s = reproduce_sample(dataset, idx)
        pred_first, pred_second = predict_patch(model, s["anchor_patch"], device)
        pred_first_pos, pred_second_pos = predict_patch(model, s["positive_patch"], device)

        A = s["transform_matrix"]
        gt_first_pos = A @ s["gt_first"]
        gt_second_pos = A @ s["gt_second"]

        first_cos, first_angle = cosine_and_angle(pred_first, s["gt_first"])
        second_cos, second_angle = cosine_and_angle(pred_second, s["gt_second"])
        spacing = patch_spacing_metrics(s["anchor_patch"])

        rows.append(
            SampleDiagnostics(
                index=idx,
                family=s["family"],
                anchor_center_index=s["anchor_center_index"],
                patch=s["anchor_patch"],
                positive_patch=s["positive_patch"],
                full_curve_centered=s["curve_centered"],
                full_curve_transformed_centered=s["curve_transformed_centered"],
                gt_first=s["gt_first"],
                gt_second=s["gt_second"],
                pred_first=pred_first,
                pred_second=pred_second,
                gt_first_pos=gt_first_pos.astype(np.float32),
                gt_second_pos=gt_second_pos.astype(np.float32),
                pred_first_pos=pred_first_pos,
                pred_second_pos=pred_second_pos,
                transform_matrix=A,
                first_cos=first_cos,
                first_angle_deg=first_angle,
                first_rmse=float(np.linalg.norm(pred_first - s["gt_first"])),
                second_cos=second_cos,
                second_angle_deg=second_angle,
                second_rmse=float(np.linalg.norm(pred_second - s["gt_second"])),
                second_gt_norm=float(np.linalg.norm(s["gt_second"])),
                pred_first_norm=float(np.linalg.norm(pred_first)),
                pred_second_norm=float(np.linalg.norm(pred_second)),
                **spacing,
            )
        )
    return rows


def records_to_dicts(rows: list[SampleDiagnostics]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        d = {
            "index": r.index,
            "family": r.family,
            "anchor_center_index": r.anchor_center_index,
            "first_cos": r.first_cos,
            "first_angle_deg": r.first_angle_deg,
            "first_rmse": r.first_rmse,
            "second_cos": r.second_cos,
            "second_angle_deg": r.second_angle_deg,
            "second_rmse": r.second_rmse,
            "second_gt_norm": r.second_gt_norm,
            "pred_first_norm": r.pred_first_norm,
            "pred_second_norm": r.pred_second_norm,
            "mean_step": r.mean_step,
            "median_step": r.median_step,
            "patch_arc": r.patch_arc,
            "patch_chord": r.patch_chord,
            "spacing_std": r.spacing_std,
            "spacing_cv": r.spacing_cv,
            "gap_ratio": r.gap_ratio,
        }
        out.append(d)
    return out


def summarize(rows: list[SampleDiagnostics], high_curv_threshold: float) -> dict[str, Any]:
    def stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.quantile(arr, 0.1)),
            "p90": float(np.quantile(arr, 0.9)),
        }

    second_norm = np.asarray([r.second_gt_norm for r in rows], dtype=np.float64)
    hi = second_norm >= high_curv_threshold

    first_cos = np.asarray([r.first_cos for r in rows], dtype=np.float64)
    first_ang = np.asarray([r.first_angle_deg for r in rows], dtype=np.float64)
    first_rmse = np.asarray([r.first_rmse for r in rows], dtype=np.float64)
    second_cos = np.asarray([r.second_cos for r in rows], dtype=np.float64)
    second_ang = np.asarray([r.second_angle_deg for r in rows], dtype=np.float64)
    second_rmse = np.asarray([r.second_rmse for r in rows], dtype=np.float64)

    summary = {
        "num_samples": len(rows),
        "high_curvature_threshold": float(high_curv_threshold),
        "high_curvature_fraction": float(hi.mean()),
        "first": {
            "cos": stats(first_cos),
            "angle_deg": stats(first_ang),
            "rmse": stats(first_rmse),
        },
        "second_all": {
            "cos": stats(second_cos),
            "angle_deg": stats(second_ang),
            "rmse": stats(second_rmse),
        },
        "second_high_curvature_only": {
            "num_samples": int(hi.sum()),
            "cos": stats(second_cos[hi]) if np.any(hi) else None,
            "angle_deg": stats(second_ang[hi]) if np.any(hi) else None,
            "rmse": stats(second_rmse[hi]) if np.any(hi) else None,
        },
        "targets": {
            "second_gt_norm": stats(second_norm),
        },
    }

    # Curvature buckets by quartiles
    qs = np.quantile(second_norm, [0.25, 0.5, 0.75]).tolist()
    edges = [-np.inf] + qs + [np.inf]
    labels = [
        f"Q1 <= {qs[0]:.3f}",
        f"Q2 ({qs[0]:.3f}, {qs[1]:.3f}]",
        f"Q3 ({qs[1]:.3f}, {qs[2]:.3f}]",
        f"Q4 > {qs[2]:.3f}",
    ]
    buckets = []
    for i in range(4):
        mask = (second_norm > edges[i]) & (second_norm <= edges[i + 1])
        buckets.append({
            "label": labels[i],
            "count": int(mask.sum()),
            "second_cos_mean": float(second_cos[mask].mean()) if np.any(mask) else None,
            "second_angle_mean": float(second_ang[mask].mean()) if np.any(mask) else None,
            "second_rmse_mean": float(second_rmse[mask].mean()) if np.any(mask) else None,
        })
    summary["curvature_quartile_buckets"] = buckets
    return summary


def write_csv(rows: list[SampleDiagnostics], path: Path) -> None:
    dict_rows = records_to_dicts(rows)
    if not dict_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        w.writeheader()
        w.writerows(dict_rows)


def _plot_metric(ax: plt.Axes, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    ax.scatter(x, y, s=10, alpha=0.18)
    if len(x) >= 20:
        quantiles = np.quantile(x, np.linspace(0.0, 1.0, 9))
        bins = np.unique(quantiles)
        if len(bins) >= 3:
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (x >= lo) & (x <= hi if hi == bins[-1] else x < hi)
                if mask.sum() >= 5:
                    xs.append(float(np.median(x[mask])))
                    ys.append(float(np.mean(y[mask])))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def plot_results_vs_metrics(rows: list[SampleDiagnostics], output_dir: Path) -> None:
    x_specs = [
        ("second_gt_norm", "GT curvature magnitude"),
        ("mean_step", "Mean consecutive patch distance"),
        ("patch_arc", "Patch arc length"),
        ("spacing_cv", "Patch spacing CV"),
    ]
    y_first = [
        ("first_angle_deg", "First angle error (deg)"),
        ("first_rmse", "First RMSE"),
    ]
    y_second = [
        ("second_angle_deg", "Second angle error (deg)"),
        ("second_rmse", "Second RMSE"),
        ("second_cos", "Second cosine"),
    ]

    dict_rows = records_to_dicts(rows)
    arrs = {k: np.asarray([r[k] for r in dict_rows], dtype=np.float64) for k in dict_rows[0].keys() if isinstance(dict_rows[0][k], (int, float))}

    fig1, axes1 = plt.subplots(len(y_first), len(x_specs), figsize=(5.0 * len(x_specs), 4.0 * len(y_first)))
    axes1 = np.atleast_2d(axes1)
    for i, (yk, yl) in enumerate(y_first):
        for j, (xk, xl) in enumerate(x_specs):
            _plot_metric(axes1[i, j], arrs[xk], arrs[yk], xl, yl, f"{yl} vs {xl}")
    fig1.tight_layout()
    fig1.savefig(output_dir / "first_derivative_vs_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(len(y_second), len(x_specs), figsize=(5.0 * len(x_specs), 4.0 * len(y_second)))
    axes2 = np.atleast_2d(axes2)
    for i, (yk, yl) in enumerate(y_second):
        for j, (xk, xl) in enumerate(x_specs):
            _plot_metric(axes2[i, j], arrs[xk], arrs[yk], xl, yl, f"{yl} vs {xl}")
    fig2.tight_layout()
    fig2.savefig(output_dir / "second_derivative_vs_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def plot_vector(ax: plt.Axes, vec: np.ndarray, color: str, label: str, scale: float = 0.35) -> None:
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return
    v = vec / n * scale
    ax.arrow(0.0, 0.0, v[0], v[1], width=0.004, head_width=0.04, head_length=0.05,
             length_includes_head=True, color=color, alpha=0.95, label=label)


def choose_visualization_indices(rows: list[SampleDiagnostics], num_visualizations: int, high_curv_threshold: float) -> list[int]:
    curv = np.asarray([r.second_gt_norm for r in rows])
    ang = np.asarray([r.second_angle_deg for r in rows])
    hi = np.where(curv >= high_curv_threshold)[0]
    lo = np.where(curv < high_curv_threshold)[0]
    chosen: list[int] = []
    if len(hi) > 0:
        chosen.append(int(hi[np.argmin(ang[hi])]))
        chosen.append(int(hi[np.argmax(ang[hi])]))
        chosen.append(int(hi[np.argsort(ang[hi])[len(hi)//2]]))
    if len(lo) > 0:
        chosen.append(int(lo[np.argmin(ang[lo])]))
        chosen.append(int(lo[np.argmax(ang[lo])]))
    # fill with evenly spaced over sorted angle
    order = np.argsort(ang)
    if len(chosen) < num_visualizations:
        pos = np.linspace(0, len(order)-1, num_visualizations).astype(int)
        for p in pos:
            idx = int(order[p])
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= num_visualizations:
                break
    return chosen[:num_visualizations]


def visualize_examples(rows: list[SampleDiagnostics], output_path: Path, num_visualizations: int, high_curv_threshold: float) -> None:
    idxs = choose_visualization_indices(rows, num_visualizations, high_curv_threshold)
    n = len(idxs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.8 * n))
    if n == 1:
        axes = np.array([axes])

    for r_idx, sample_idx in enumerate(idxs):
        s = rows[sample_idx]
        ax0, ax1 = axes[r_idx]

        # Original
        ax0.plot(s.full_curve_centered[:, 0], s.full_curve_centered[:, 1], color="0.8", linewidth=1.5)
        ax0.plot(s.patch[:, 0], s.patch[:, 1], color="black", marker="o", markersize=3.0, linewidth=2.0)
        ax0.scatter([0.0], [0.0], color="black", s=24)
        plot_vector(ax0, s.gt_first, "tab:green", "GT first")
        plot_vector(ax0, s.pred_first, "tab:red", "Pred first")
        plot_vector(ax0, s.gt_second, "tab:blue", "GT second")
        plot_vector(ax0, s.pred_second, "tab:orange", "Pred second")
        ax0.set_aspect("equal")
        ax0.grid(True, alpha=0.2)
        ax0.set_title(
            f"Original | idx={s.index}, curv={s.second_gt_norm:.2f}\n"
            f"first={s.first_angle_deg:.1f}°, second={s.second_angle_deg:.1f}°, mean_step={s.mean_step:.3f}",
            fontsize=10,
        )
        h, l = ax0.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        ax0.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

        # Transformed
        ax1.plot(s.full_curve_transformed_centered[:, 0], s.full_curve_transformed_centered[:, 1], color="0.8", linewidth=1.5)
        ax1.plot(s.positive_patch[:, 0], s.positive_patch[:, 1], color="black", marker="o", markersize=3.0, linewidth=2.0)
        ax1.scatter([0.0], [0.0], color="black", s=24)
        plot_vector(ax1, s.gt_first_pos, "tab:green", "GT first (+)")
        plot_vector(ax1, s.pred_first_pos, "tab:red", "Pred first (+)")
        plot_vector(ax1, s.gt_second_pos, "tab:blue", "GT second (+)")
        plot_vector(ax1, s.pred_second_pos, "tab:orange", "Pred second (+)")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.2)
        ax1.set_title("Transformed counterpart", fontsize=10)
        h, l = ax1.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        ax1.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    fig.suptitle("Test-set qualitative diagnostics: original and transformed examples", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_text_summary(summary: dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("Operator diagnostics summary")
    lines.append("=" * 28)
    lines.append(f"num_samples: {summary['num_samples']}")
    lines.append(f"high_curvature_threshold: {summary['high_curvature_threshold']:.4f}")
    lines.append(f"high_curvature_fraction: {summary['high_curvature_fraction']:.4f}")
    lines.append("")
    lines.append("First derivative")
    lines.append(f"  mean cosine: {summary['first']['cos']['mean']:.4f}")
    lines.append(f"  mean angle:  {summary['first']['angle_deg']['mean']:.3f} deg")
    lines.append(f"  mean rmse:   {summary['first']['rmse']['mean']:.4f}")
    lines.append("")
    lines.append("Second derivative (all test samples)")
    lines.append(f"  mean cosine: {summary['second_all']['cos']['mean']:.4f}")
    lines.append(f"  mean angle:  {summary['second_all']['angle_deg']['mean']:.3f} deg")
    lines.append(f"  mean rmse:   {summary['second_all']['rmse']['mean']:.4f}")
    lines.append("")
    hi = summary['second_high_curvature_only']
    lines.append("Second derivative (high curvature only)")
    lines.append(f"  count:       {hi['num_samples']}")
    if hi['cos'] is not None:
        lines.append(f"  mean cosine: {hi['cos']['mean']:.4f}")
        lines.append(f"  mean angle:  {hi['angle_deg']['mean']:.3f} deg")
        lines.append(f"  mean rmse:   {hi['rmse']['mean']:.4f}")
    lines.append("")
    lines.append("Curvature quartile buckets")
    for bucket in summary["curvature_quartile_buckets"]:
        lines.append(
            f"  {bucket['label']}: count={bucket['count']}, "
            f"second_cos_mean={bucket['second_cos_mean']}, "
            f"second_angle_mean={bucket['second_angle_mean']}, "
            f"second_rmse_mean={bucket['second_rmse_mean']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)

    dataset = make_test_dataset(args)
    model = build_model(device)
    load_checkpoint(model, checkpoint_path, device)

    rows = evaluate_dataset(model, dataset, device)
    summary = summarize(rows, args.high_curvature_threshold)

    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_text_summary(summary, output_dir / "diagnostic_summary.txt")
    write_csv(rows, output_dir / "per_sample_metrics.csv")
    plot_results_vs_metrics(rows, output_dir)
    visualize_examples(
        rows,
        output_dir / "qualitative_examples.png",
        args.num_visualizations,
        args.high_curvature_threshold,
    )

    print(f"Wrote diagnostics to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
