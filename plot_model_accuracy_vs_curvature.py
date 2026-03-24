# plot_model_accuracy_vs_curvature.py
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from models.tangent_model import TangentOperatorModel


def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine(a, b):
    a = normalize(a)
    b = normalize(b)
    return (a * b).sum(dim=-1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--operator-kernel-size", type=int, default=5)
    parser.add_argument("--test-length", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-mode", type=str, default="random_warp_symmetric")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # dataset
    ds = TangentDataset(
        length=args.test_length,
        family="euclidean",
        patch_size=args.patch_size,
        patch_mode=args.patch_mode,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tangent_collate_fn,
    )

    # model
    model = TangentOperatorModel(
        patch_size=args.patch_size,
        operator_kernel_size=args.operator_kernel_size,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_cos = []
    all_curv = []

    for batch in loader:
        x = batch.anchor.to(device)
        out = model(x)

        pred_second = out["vector_second"]
        gt_second = batch.gt_second_anchor.to(device)

        cos_val = cosine(pred_second, gt_second).cpu()
        curv = gt_second.norm(dim=-1).cpu()

        all_cos.append(cos_val)
        all_curv.append(curv)

    cos_all = torch.cat(all_cos).numpy()
    curv_all = torch.cat(all_curv).numpy()

    # ---------- Scatter ----------
    plt.figure(figsize=(6, 5))
    plt.scatter(curv_all, cos_all, s=3, alpha=0.3)
    plt.xlabel("||GT second derivative|| (curvature proxy)")
    plt.ylabel("cos(pred_second, gt_second)")
    plt.title("Model accuracy vs curvature (scatter)")
    plt.grid(True)
    plt.savefig(out_dir / "scatter_accuracy_vs_curvature.png", dpi=200)
    plt.close()

    # ---------- Binned ----------
    num_bins = 20
    bins = np.linspace(curv_all.min(), curv_all.max(), num_bins + 1)
    bin_centers = []
    bin_means = []

    for i in range(num_bins):
        mask = (curv_all >= bins[i]) & (curv_all < bins[i + 1])
        if mask.sum() < 10:
            continue
        bin_centers.append(curv_all[mask].mean())
        bin_means.append(cos_all[mask].mean())

    plt.figure(figsize=(6, 5))
    plt.plot(bin_centers, bin_means, marker="o")
    plt.xlabel("Mean ||GT second derivative|| per bin")
    plt.ylabel("Mean cosine accuracy")
    plt.title("Binned accuracy vs curvature")
    plt.grid(True)
    plt.savefig(out_dir / "binned_accuracy_vs_curvature.png", dpi=200)
    plt.close()

    # ---------- Save summary ----------
    summary = {
        "mean_cosine": float(np.mean(cos_all)),
        "median_cosine": float(np.median(cos_all)),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
