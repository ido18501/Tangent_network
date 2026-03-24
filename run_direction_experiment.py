from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Allow running either from project_root/scripts/ or directly from project_root.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT_CANDIDATES = [THIS_FILE.parent, THIS_FILE.parent.parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn, TangentBatch
from training.trainer import TangentTrainer


class DirectionOnlyOperatorLoss(nn.Module):
    def __init__(
        self,
        lambda_reg: float = 1e-4,
        lambda_neg: float = 0.1,
        neg_cos_margin: float = 0.2,
        lambda_first: float = 1.0,
        lambda_second: float = 1.0,
        lambda_equiv_first: float = 1.0,
        lambda_equiv_second: float = 1.0,
        direction_eps: float = 1e-8,
        second_dir_min_norm: float = 1e-2,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_neg = lambda_neg
        self.neg_cos_margin = neg_cos_margin
        self.lambda_first = lambda_first
        self.lambda_second = lambda_second
        self.lambda_equiv_first = lambda_equiv_first
        self.lambda_equiv_second = lambda_equiv_second
        self.direction_eps = direction_eps
        self.second_dir_min_norm = second_dir_min_norm

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(self.direction_eps)

    def _signed_direction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_dir = self._normalize(pred)
        target_dir = self._normalize(target)
        cos_sim = (pred_dir * target_dir).sum(dim=-1)
        loss_per_sample = 1.0 - cos_sim

        if mask is not None:
            mask = mask.to(dtype=loss_per_sample.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (loss_per_sample * mask).sum() / denom
        return loss_per_sample.mean()

    def _negative_direction_loss(
        self,
        pred_negatives: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        neg_dir = self._normalize(pred_negatives)
        target_dir = self._normalize(target).unsqueeze(1)
        cos_sim = (neg_dir * target_dir).sum(dim=-1)
        return F.relu(cos_sim - self.neg_cos_margin).mean()

    def forward(
        self,
        *,
        v_first_anchor: torch.Tensor,
        v_first_positive: torch.Tensor,
        v_second_anchor: torch.Tensor,
        v_second_positive: torch.Tensor,
        weights_first_anchor: torch.Tensor,
        weights_second_anchor: torch.Tensor,
        transform_matrix: torch.Tensor,
        gt_first_anchor: torch.Tensor,
        gt_second_anchor: torch.Tensor,
        v_first_negatives: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        target_first = torch.einsum("bij,bj->bi", transform_matrix, v_first_anchor)
        target_second = torch.einsum("bij,bj->bi", transform_matrix, v_second_anchor)

        gt_second_norm = gt_second_anchor.norm(dim=-1)
        second_mask = gt_second_norm > self.second_dir_min_norm

        equiv_first_loss = self._signed_direction_loss(v_first_positive, target_first)
        equiv_second_loss = self._signed_direction_loss(v_second_positive, target_second, mask=second_mask)
        first_loss = self._signed_direction_loss(v_first_anchor, gt_first_anchor)
        second_loss = self._signed_direction_loss(v_second_anchor, gt_second_anchor, mask=second_mask)
        reg_loss = weights_first_anchor.pow(2).mean()

        neg_loss = torch.tensor(0.0, device=v_first_anchor.device, dtype=v_first_anchor.dtype)
        if v_first_negatives is not None and self.lambda_neg > 0.0:
            neg_loss = self._negative_direction_loss(v_first_negatives, gt_first_anchor)

        loss = (
            self.lambda_equiv_first * equiv_first_loss
            + self.lambda_equiv_second * equiv_second_loss
            + self.lambda_first * first_loss
            + self.lambda_second * second_loss
            + self.lambda_neg * neg_loss
            + self.lambda_reg * reg_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            first_anchor_dir = self._normalize(v_first_anchor)
            gt_first_dir = self._normalize(gt_first_anchor)
            first_cos = (first_anchor_dir * gt_first_dir).sum(dim=-1)

            second_anchor_dir = self._normalize(v_second_anchor)
            gt_second_dir = self._normalize(gt_second_anchor)
            second_cos = (second_anchor_dir * gt_second_dir).sum(dim=-1)

            stats = {
                "loss": float(loss.detach().item()),
                "equiv_first_loss": float(equiv_first_loss.detach().item()),
                "equiv_second_loss": float(equiv_second_loss.detach().item()),
                "first_loss": float(first_loss.detach().item()),
                "second_loss": float(second_loss.detach().item()),
                "neg_loss": float(neg_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
                "first_vector_norm_mean": float(v_first_anchor.norm(dim=-1).mean().detach().item()),
                "second_vector_norm_mean": float(v_second_anchor.norm(dim=-1).mean().detach().item()),
                "gt_first_norm_mean": float(gt_first_anchor.norm(dim=-1).mean().detach().item()),
                "gt_second_norm_mean": float(gt_second_anchor.norm(dim=-1).mean().detach().item()),
                "first_direction_cos_mean": float(first_cos.mean().detach().item()),
                "second_direction_cos_mean": float(second_cos[second_mask].mean().detach().item()) if second_mask.any() else 0.0,
                "second_mask_fraction": float(second_mask.float().mean().detach().item()),
                "weight_first_sum_mean": float(weights_first_anchor.sum(dim=-1).mean().detach().item()),
                "weight_second_sum_mean": float(weights_second_anchor.sum(dim=-1).mean().detach().item()),
            }
        return loss, stats


@dataclass
class PredictionRecord:
    patch: np.ndarray
    gt_first: np.ndarray
    pred_first: np.ndarray
    gt_second: np.ndarray
    pred_second: np.ndarray
    first_cos: float
    first_angle_deg: float
    second_cos: float
    second_angle_deg: float
    second_gt_norm: float
    second_masked: bool
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the tangent operator with direction-only losses.")
    parser.add_argument("--output-dir", type=str, default="runs/direction_experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--train-length", type=int, default=12000)
    parser.add_argument("--val-length", type=int, default=2000)
    parser.add_argument("--test-length", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--transform-family", type=str, default="euclidean", choices=["euclidean", "similarity", "equi_affine", "affine"])
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--half-width", type=int, default=12)
    parser.add_argument("--num-negatives", type=int, default=8)
    parser.add_argument("--negative-min-offset", type=int, default=5)
    parser.add_argument("--negative-max-offset", type=int, default=25)
    parser.add_argument("--negative-other-curve-fraction", type=float, default=0.5)
    parser.add_argument("--num-curve-points", type=int, default=300)
    parser.add_argument("--patch-mode", type=str, default="random_warp_symmetric", choices=["uniform_symmetric", "random_warp_symmetric"])
    parser.add_argument("--jitter-fraction", type=float, default=0.25)
    parser.add_argument("--closed", action="store_true", default=True)
    parser.add_argument("--no-closed", dest="closed", action="store_false")
    parser.add_argument("--warp-sampling-prob", type=float, default=0.7)
    parser.add_argument("--warp-sampling-strength", type=float, default=0.18)
    parser.add_argument("--point-noise-std", type=float, default=0.0)
    parser.add_argument("--orthogonal-noise-std", type=float, default=0.0)
    parser.add_argument("--gt-dense-num-points", type=int, default=4096)
    parser.add_argument("--family-probs", type=str, default="fourier:1.0,piecewise:0.0")

    parser.add_argument("--point-mlp-dims", type=str, default="64,64,128")
    parser.add_argument("--head-dims", type=str, default="128,64")
    parser.add_argument("--use-batchnorm", action="store_true", default=True)
    parser.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false")
    parser.add_argument("--point-dropout", type=float, default=0.0)
    parser.add_argument("--head-dropout", type=float, default=0.0)

    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--lambda-neg", type=float, default=0.1)
    parser.add_argument("--neg-cos-margin", type=float, default=0.2)
    parser.add_argument("--lambda-first", type=float, default=1.0)
    parser.add_argument("--lambda-second", type=float, default=1.0)
    parser.add_argument("--lambda-equiv-first", type=float, default=1.0)
    parser.add_argument("--lambda-equiv-second", type=float, default=1.0)
    parser.add_argument("--second-dir-min-norm", type=float, default=1e-2)

    parser.add_argument("--num-viz-examples", type=int, default=9)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_family_probs(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in text.split(","):
        name, value = item.split(":")
        out[name.strip()] = float(value.strip())
    if not out:
        raise ValueError("family_probs is empty")
    total = sum(out.values())
    if total <= 0:
        raise ValueError("family_probs must sum to a positive number")
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(args: argparse.Namespace, length: int, seed: int) -> TangentDataset:
    return TangentDataset(
        length=length,
        family=args.transform_family,
        num_curve_points=args.num_curve_points,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        negative_min_offset=args.negative_min_offset,
        negative_max_offset=args.negative_max_offset,
        negative_other_curve_fraction=args.negative_other_curve_fraction,
        patch_mode=args.patch_mode,
        jitter_fraction=args.jitter_fraction,
        closed=args.closed,
        return_centered=True,
        point_noise_std=args.point_noise_std,
        curve_family_probs=parse_family_probs(args.family_probs),
        warp_sampling_prob=args.warp_sampling_prob,
        warp_sampling_strength=args.warp_sampling_strength,
        orthogonal_noise_std=args.orthogonal_noise_std,
        gt_dense_num_points=args.gt_dense_num_points,
        seed=seed,
    )


def make_loader(dataset: TangentDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tangent_collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def average_epoch_outputs(outputs: list[dict[str, float]]) -> dict[str, float]:
    if not outputs:
        return {}
    keys = sorted(outputs[0].keys())
    return {k: float(np.mean([o[k] for o in outputs])) for k in keys}


def run_epoch(trainer: TangentTrainer, loader: DataLoader, train: bool) -> dict[str, float]:
    outputs: list[dict[str, float]] = []
    for batch in loader:
        out = trainer.train_step(batch) if train else trainer.eval_step(batch)
        outputs.append(out.stats)
    return average_epoch_outputs(outputs)


def cosine_and_angle(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_n = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    return cos, angle


@torch.no_grad()
def collect_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    second_dir_min_norm: float,
) -> list[PredictionRecord]:
    model.eval()
    records: list[PredictionRecord] = []
    for batch in loader:
        batch = TangentBatch(
            anchor=batch.anchor.to(device),
            positive=batch.positive.to(device),
            negatives=batch.negatives.to(device),
            transform_matrix=batch.transform_matrix.to(device),
            family=batch.family,
            anchor_center_index=batch.anchor_center_index,
            negative_center_indices=batch.negative_center_indices,
            gt_first_anchor=batch.gt_first_anchor.to(device),
            gt_second_anchor=batch.gt_second_anchor.to(device),
        )
        out = model(batch.anchor)
        pred_first = out["vector_first"]
        pred_second = out["vector_second"]
        gt_first = batch.gt_first_anchor
        gt_second = batch.gt_second_anchor

        first_cos, first_angle = cosine_and_angle(pred_first, gt_first)
        second_cos, second_angle = cosine_and_angle(pred_second, gt_second)
        second_norm = gt_second.norm(dim=-1)
        second_mask = second_norm > second_dir_min_norm

        for i in range(batch.anchor.shape[0]):
            records.append(
                PredictionRecord(
                    patch=batch.anchor[i].detach().cpu().numpy(),
                    gt_first=gt_first[i].detach().cpu().numpy(),
                    pred_first=pred_first[i].detach().cpu().numpy(),
                    gt_second=gt_second[i].detach().cpu().numpy(),
                    pred_second=pred_second[i].detach().cpu().numpy(),
                    first_cos=float(first_cos[i].detach().cpu().item()),
                    first_angle_deg=float(first_angle[i].detach().cpu().item()),
                    second_cos=float(second_cos[i].detach().cpu().item()),
                    second_angle_deg=float(second_angle[i].detach().cpu().item()),
                    second_gt_norm=float(second_norm[i].detach().cpu().item()),
                    second_masked=bool(second_mask[i].detach().cpu().item()),
                    family=batch.family[i],
                )
            )
    return records


def summarize_predictions(records: list[PredictionRecord]) -> dict[str, Any]:
    if not records:
        return {}

    first_angles = np.asarray([r.first_angle_deg for r in records], dtype=np.float64)
    first_cos = np.asarray([r.first_cos for r in records], dtype=np.float64)
    second_angles_all = np.asarray([r.second_angle_deg for r in records], dtype=np.float64)
    second_cos_all = np.asarray([r.second_cos for r in records], dtype=np.float64)
    second_norms = np.asarray([r.second_gt_norm for r in records], dtype=np.float64)
    second_valid = np.asarray([r.second_masked for r in records], dtype=bool)

    summary: dict[str, Any] = {
        "num_test_samples": int(len(records)),
        "first": {
            "mean_angle_deg": float(first_angles.mean()),
            "median_angle_deg": float(np.median(first_angles)),
            "p90_angle_deg": float(np.quantile(first_angles, 0.9)),
            "mean_cos": float(first_cos.mean()),
            "acc_le_5deg": float(np.mean(first_angles <= 5.0)),
            "acc_le_10deg": float(np.mean(first_angles <= 10.0)),
            "acc_le_20deg": float(np.mean(first_angles <= 20.0)),
        },
        "second_all": {
            "mean_angle_deg": float(second_angles_all.mean()),
            "median_angle_deg": float(np.median(second_angles_all)),
            "p90_angle_deg": float(np.quantile(second_angles_all, 0.9)),
            "mean_cos": float(second_cos_all.mean()),
        },
        "second_valid_fraction": float(second_valid.mean()),
        "second_gt_norm": {
            "mean": float(second_norms.mean()),
            "median": float(np.median(second_norms)),
            "p90": float(np.quantile(second_norms, 0.9)),
        },
    }

    if np.any(second_valid):
        second_angles = second_angles_all[second_valid]
        second_cos = second_cos_all[second_valid]
        summary["second_valid_only"] = {
            "num_samples": int(second_valid.sum()),
            "mean_angle_deg": float(second_angles.mean()),
            "median_angle_deg": float(np.median(second_angles)),
            "p90_angle_deg": float(np.quantile(second_angles, 0.9)),
            "mean_cos": float(second_cos.mean()),
            "acc_le_10deg": float(np.mean(second_angles <= 10.0)),
            "acc_le_20deg": float(np.mean(second_angles <= 20.0)),
            "acc_le_30deg": float(np.mean(second_angles <= 30.0)),
        }
    else:
        summary["second_valid_only"] = {
            "num_samples": 0,
        }

    return summary


def format_summary_text(summary: dict[str, Any]) -> str:
    if not summary:
        return "No test records collected.\n"

    lines = []
    lines.append("Direction-only tangent-operator test summary")
    lines.append("=" * 44)
    lines.append(f"num_test_samples: {summary['num_test_samples']}")
    lines.append("")
    lines.append("First derivative direction")
    lines.append(f"  mean angle (deg):   {summary['first']['mean_angle_deg']:.3f}")
    lines.append(f"  median angle (deg): {summary['first']['median_angle_deg']:.3f}")
    lines.append(f"  p90 angle (deg):    {summary['first']['p90_angle_deg']:.3f}")
    lines.append(f"  mean cosine:        {summary['first']['mean_cos']:.4f}")
    lines.append(f"  <= 5 deg:           {summary['first']['acc_le_5deg']:.4f}")
    lines.append(f"  <= 10 deg:          {summary['first']['acc_le_10deg']:.4f}")
    lines.append(f"  <= 20 deg:          {summary['first']['acc_le_20deg']:.4f}")
    lines.append("")
    lines.append("Second derivative direction (all test samples)")
    lines.append(f"  mean angle (deg):   {summary['second_all']['mean_angle_deg']:.3f}")
    lines.append(f"  median angle (deg): {summary['second_all']['median_angle_deg']:.3f}")
    lines.append(f"  p90 angle (deg):    {summary['second_all']['p90_angle_deg']:.3f}")
    lines.append(f"  mean cosine:        {summary['second_all']['mean_cos']:.4f}")
    lines.append(f"  valid fraction:     {summary['second_valid_fraction']:.4f}")
    lines.append("")
    lines.append("Second derivative direction (valid curvature only)")
    lines.append(f"  num samples:        {summary['second_valid_only'].get('num_samples', 0)}")
    if summary['second_valid_only'].get('num_samples', 0) > 0:
        lines.append(f"  mean angle (deg):   {summary['second_valid_only']['mean_angle_deg']:.3f}")
        lines.append(f"  median angle (deg): {summary['second_valid_only']['median_angle_deg']:.3f}")
        lines.append(f"  p90 angle (deg):    {summary['second_valid_only']['p90_angle_deg']:.3f}")
        lines.append(f"  mean cosine:        {summary['second_valid_only']['mean_cos']:.4f}")
        lines.append(f"  <= 10 deg:          {summary['second_valid_only']['acc_le_10deg']:.4f}")
        lines.append(f"  <= 20 deg:          {summary['second_valid_only']['acc_le_20deg']:.4f}")
        lines.append(f"  <= 30 deg:          {summary['second_valid_only']['acc_le_30deg']:.4f}")
    return "\n".join(lines) + "\n"


def choose_example_indices(records: list[PredictionRecord], num_examples: int) -> list[int]:
    if len(records) <= num_examples:
        return list(range(len(records)))

    scores = []
    for i, r in enumerate(records):
        second_part = r.second_angle_deg if r.second_masked else 90.0
        score = 0.5 * r.first_angle_deg + 0.5 * second_part
        scores.append((score, i))
    scores.sort()

    positions = np.linspace(0, len(scores) - 1, num_examples)
    chosen = sorted({scores[int(round(pos))][1] for pos in positions})
    if len(chosen) < num_examples:
        for _, idx in scores:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == num_examples:
                break
    return chosen[:num_examples]


def plot_vector(ax, origin: np.ndarray, vec: np.ndarray, color: str, label: str, scale: float = 0.35, linewidth: float = 2.5) -> None:
    v = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return
    v = (v / norm) * scale
    ax.arrow(
        origin[0],
        origin[1],
        v[0],
        v[1],
        width=0.004,
        head_width=0.04,
        head_length=0.05,
        length_includes_head=True,
        color=color,
        linewidth=linewidth,
        label=label,
        alpha=0.95,
    )


def visualize_examples(records: list[PredictionRecord], output_path: Path, num_examples: int) -> None:
    indices = choose_example_indices(records, num_examples)
    rows = int(math.ceil(len(indices) / 3))
    cols = min(3, len(indices))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.0 * rows))
    axes_arr = np.array(axes, dtype=object).reshape(rows, cols)

    for ax in axes_arr.flat:
        ax.axis("off")

    for ax, idx in zip(axes_arr.flat, indices):
        r = records[idx]
        ax.axis("on")
        patch = r.patch
        ax.plot(patch[:, 0], patch[:, 1], marker="o", markersize=3.0, linewidth=1.5, color="0.55")
        ax.scatter([0.0], [0.0], color="black", s=30, zorder=5)
        plot_vector(ax, np.zeros(2), r.gt_first, color="tab:green", label="GT first")
        plot_vector(ax, np.zeros(2), r.pred_first, color="tab:red", label="Pred first")
        plot_vector(ax, np.zeros(2), r.gt_second, color="tab:blue", label="GT second")
        plot_vector(ax, np.zeros(2), r.pred_second, color="tab:orange", label="Pred second")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f"family={r.family}\n"
            f"first={r.first_angle_deg:.1f}°, second={r.second_angle_deg:.1f}°"
            f"{' (masked)' if not r.second_masked else ''}",
            fontsize=10,
        )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    fig.suptitle("Test examples: patch geometry with GT/predicted first and second directions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_cos1 = [h.get("train_first_direction_cos_mean", float("nan")) for h in history]
    val_cos1 = [h.get("val_first_direction_cos_mean", float("nan")) for h in history]
    train_cos2 = [h.get("train_second_direction_cos_mean", float("nan")) for h in history]
    val_cos2 = [h.get("val_second_direction_cos_mean", float("nan")) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, train_cos1, label="train")
    axes[1].plot(epochs, val_cos1, label="val")
    axes[1].set_title("First-direction cosine")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    axes[2].plot(epochs, train_cos2, label="train")
    axes[2].plot(epochs, val_cos2, label="val")
    axes[2].set_title("Second-direction cosine")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_histograms(records: list[PredictionRecord], output_path: Path) -> None:
    first_angles = np.asarray([r.first_angle_deg for r in records], dtype=np.float64)
    second_valid_angles = np.asarray([r.second_angle_deg for r in records if r.second_masked], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(first_angles, bins=30)
    axes[0].set_title("First-derivative angle error")
    axes[0].set_xlabel("degrees")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.2)

    if len(second_valid_angles) > 0:
        axes[1].hist(second_valid_angles, bins=30)
        axes[1].set_title("Second-derivative angle error (valid only)")
        axes[1].set_xlabel("degrees")
        axes[1].set_ylabel("count")
        axes[1].grid(True, alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, "No valid second-derivative samples", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    keys = sorted(history[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_dataset = make_dataset(args, args.train_length, args.seed)
    val_dataset = make_dataset(args, args.val_length, args.seed + 10_000)
    test_dataset = make_dataset(args, args.test_length, args.seed + 20_000)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, args.seed + 1)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, args.seed + 2)

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        point_mlp_dims=parse_int_list(args.point_mlp_dims),
        head_dims=parse_int_list(args.head_dims),
        use_batchnorm=args.use_batchnorm,
        point_dropout=args.point_dropout,
        head_dropout=args.head_dropout,
    )

    loss_fn = DirectionOnlyOperatorLoss(
        lambda_reg=args.lambda_reg,
        lambda_neg=args.lambda_neg,
        neg_cos_margin=args.neg_cos_margin,
        lambda_first=args.lambda_first,
        lambda_second=args.lambda_second,
        lambda_equiv_first=args.lambda_equiv_first,
        lambda_equiv_second=args.lambda_equiv_second,
        second_dir_min_norm=args.second_dir_min_norm,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=checkpoint_dir,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    best_model_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(trainer, train_loader, train=True)
        val_metrics = run_epoch(trainer, val_loader, train=False)
        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))

        row: dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        history.append(row)

        print(f"\nEpoch {epoch}")
        print("train:", train_metrics)
        print("val:  ", val_metrics)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
            print("✓ saved new best model")
        else:
            patience_count += 1
            print(f"patience: {patience_count}/{args.patience}")
            if patience_count >= args.patience:
                print("Early stopping triggered")
                break

    print(f"\nBest validation epoch: {best_epoch}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    test_records = collect_test_predictions(model, test_loader, device, args.second_dir_min_norm)
    test_summary = summarize_predictions(test_records)
    summary_text = format_summary_text(test_summary)

    plot_history(history, output_dir / "train_val_history.png")
    plot_test_histograms(test_records, output_dir / "test_angle_histograms.png")
    visualize_examples(test_records, output_dir / "test_examples.png", args.num_viz_examples)
    save_history_csv(history, output_dir / "history.csv")

    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    with (output_dir / "test_summary.txt").open("w", encoding="utf-8") as f:
        f.write(summary_text)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + summary_text)
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
