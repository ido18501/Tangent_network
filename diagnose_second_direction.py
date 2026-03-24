# diagnose_second_direction.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from models.tangent_model import TangentOperatorModel


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()

    all_pred_first = []
    all_pred_second = []
    all_gt_first = []
    all_gt_second = []

    for batch in loader:
        anchor = batch.anchor.to(device)
        out = model(anchor)

        all_pred_first.append(out["vector_first"].cpu())
        all_pred_second.append(out["vector_second"].cpu())
        all_gt_first.append(batch.gt_first_anchor.cpu())
        all_gt_second.append(batch.gt_second_anchor.cpu())

    pred_first = torch.cat(all_pred_first, dim=0)
    pred_second = torch.cat(all_pred_second, dim=0)
    gt_first = torch.cat(all_gt_first, dim=0)
    gt_second = torch.cat(all_gt_second, dim=0)

    return pred_first, pred_second, gt_first, gt_second


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = normalize(a)
    b = normalize(b)
    return (a * b).sum(dim=-1)


def summarize_bucket(name: str, mask: torch.Tensor, cos_ps_gs: torch.Tensor,
                     cos_ps_gf: torch.Tensor, cos_ps_pf: torch.Tensor,
                     gt_second_norm: torch.Tensor):
    mask = mask.bool()
    if mask.sum().item() == 0:
        return {
            "name": name,
            "count": 0,
        }

    vals = {
        "name": name,
        "count": int(mask.sum().item()),
        "gt_second_norm_mean": float(gt_second_norm[mask].mean().item()),
        "cos_pred_second_gt_second_mean": float(cos_ps_gs[mask].mean().item()),
        "abs_cos_pred_second_gt_second_mean": float(cos_ps_gs[mask].abs().mean().item()),
        "cos_pred_second_gt_first_mean": float(cos_ps_gf[mask].mean().item()),
        "abs_cos_pred_second_gt_first_mean": float(cos_ps_gf[mask].abs().mean().item()),
        "cos_pred_second_pred_first_mean": float(cos_ps_pf[mask].mean().item()),
        "abs_cos_pred_second_pred_first_mean": float(cos_ps_pf[mask].abs().mean().item()),
    }
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--transform-family", type=str, default="euclidean")
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--operator-kernel-size", type=int, default=5)
    p.add_argument("--half-width", type=int, default=12)

    p.add_argument("--test-length", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--patch-mode", type=str, default="random_warp_symmetric")
    p.add_argument("--jitter-fraction", type=float, default=0.25)
    p.add_argument("--num-negatives", type=int, default=8)
    p.add_argument("--negative-min-offset", type=int, default=20)
    p.add_argument("--negative-max-offset", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = TangentDataset(
        length=args.test_length,
        family="euclidean",
        patch_size=args.patch_size,
        half_width=args.half_width,
        patch_mode=args.patch_mode,
        jitter_fraction=args.jitter_fraction,
        num_negatives=args.num_negatives,
        negative_min_offset=args.negative_min_offset,
        negative_max_offset=args.negative_max_offset,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tangent_collate_fn,
    )

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        operator_kernel_size=args.operator_kernel_size,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    pred_first, pred_second, gt_first, gt_second = collect_outputs(model, loader, device)

    cos_ps_gs = cosine(pred_second, gt_second)
    cos_ps_gf = cosine(pred_second, gt_first)
    cos_ps_pf = cosine(pred_second, pred_first)
    gt_second_norm = gt_second.norm(dim=-1)

    n = len(gt_second_norm)
    order = torch.argsort(gt_second_norm, descending=True)

    top50 = torch.zeros(n, dtype=torch.bool)
    top25 = torch.zeros(n, dtype=torch.bool)
    top10 = torch.zeros(n, dtype=torch.bool)

    top50[order[: max(1, n // 2)]] = True
    top25[order[: max(1, n // 4)]] = True
    top10[order[: max(1, n // 10)]] = True
    all_mask = torch.ones(n, dtype=torch.bool)

    summaries = [
        summarize_bucket("all", all_mask, cos_ps_gs, cos_ps_gf, cos_ps_pf, gt_second_norm),
        summarize_bucket("top_50_percent_gt_second_norm", top50, cos_ps_gs, cos_ps_gf, cos_ps_pf, gt_second_norm),
        summarize_bucket("top_25_percent_gt_second_norm", top25, cos_ps_gs, cos_ps_gf, cos_ps_pf, gt_second_norm),
        summarize_bucket("top_10_percent_gt_second_norm", top10, cos_ps_gs, cos_ps_gf, cos_ps_pf, gt_second_norm),
    ]

    results = {
        "checkpoint": args.checkpoint,
        "patch_mode": args.patch_mode,
        "summaries": summaries,
    }

    print(json.dumps(results, indent=2))

    with open(out_dir / "second_direction_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
