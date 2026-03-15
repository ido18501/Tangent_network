from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentPatchEmbeddingModel
from training.collate import tangent_collate_fn
from training.losses import TupleInfoNCELoss


def build_dataset(args, split: str) -> TangentDataset:
    if split == "train":
        length = args.train_size
        seed = args.train_seed
    elif split == "val":
        length = args.val_size
        seed = args.val_seed
    elif split == "test":
        length = args.test_size
        seed = args.test_seed
    else:
        raise ValueError(f"Unknown split: {split}")

    return TangentDataset(
        length=length,
        family=args.family,
        num_curve_points=args.num_curve_points,
        fourier_max_freq=args.fourier_max_freq,
        fourier_scale=args.fourier_scale,
        fourier_decay_power=args.fourier_decay_power,
        curve_max_tries=args.curve_max_tries,
        curve_min_size=args.curve_min_size,
        curve_max_size=args.curve_max_size,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        negative_min_offset=args.negative_min_offset,
        negative_max_offset=args.negative_max_offset,
        patch_mode=args.patch_mode,
        jitter_fraction=args.jitter_fraction,
        closed=True,
        transform_kwargs={"allow_reflection": args.allow_reflection},
        return_centered=True,
        seed=seed,
    )


@torch.no_grad()
def embed_batch(model, batch, device):
    anchor = batch.anchor.to(device)          # (B, P, 2)
    positive = batch.positive.to(device)      # (B, P, 2)
    negatives = batch.negatives.to(device)    # (B, M, P, 2)

    B, M, P, C = negatives.shape

    a = model(anchor)                         # (B, D)
    p = model(positive)                       # (B, D)
    n = model(negatives.view(B * M, P, C))    # (B*M, D)
    n = n.view(B, M, -1)                      # (B, M, D)

    # Safety normalization
    a = F.normalize(a, p=2, dim=-1)
    p = F.normalize(p, p=2, dim=-1)
    n = F.normalize(n, p=2, dim=-1)

    return a, p, n


@torch.no_grad()
def evaluate_loader(model, loss_fn, loader, device, max_vis_samples: int = 5000):
    model.eval()

    total_weight = 0
    weighted_sums = {
        "loss": 0.0,
        "positive_similarity_mean": 0.0,
        "negative_similarity_mean": 0.0,
        "logits_mean": 0.0,
        "hardest_negative_similarity_mean": 0.0,
        "positive_minus_hardest_negative_mean": 0.0,
    }

    anchor_embs = []
    positive_embs = []
    pos_sims_all = []
    hard_neg_sims_all = []

    for batch in loader:
        a, p, n = embed_batch(model, batch, device)

        loss, stats = loss_fn(a, p, n, return_stats=True)

        pos_sim = torch.sum(a * p, dim=-1)                    # (B,)
        neg_sim = torch.einsum("bd,bmd->bm", a, n)            # (B, M)
        hard_neg_sim = neg_sim.max(dim=1).values              # (B,)

        B = a.shape[0]
        total_weight += B

        weighted_sums["loss"] += float(loss.item()) * B
        weighted_sums["positive_similarity_mean"] += float(pos_sim.mean().item()) * B
        weighted_sums["negative_similarity_mean"] += float(neg_sim.mean().item()) * B
        weighted_sums["logits_mean"] += float(stats.logits_mean) * B
        weighted_sums["hardest_negative_similarity_mean"] += float(hard_neg_sim.mean().item()) * B
        weighted_sums["positive_minus_hardest_negative_mean"] += float((pos_sim - hard_neg_sim).mean().item()) * B

        if sum(x.shape[0] for x in anchor_embs) < max_vis_samples:
            anchor_embs.append(a.cpu())
            positive_embs.append(p.cpu())

        pos_sims_all.append(pos_sim.cpu())
        hard_neg_sims_all.append(hard_neg_sim.cpu())

    metrics = {k: v / total_weight for k, v in weighted_sums.items()}

    anchor_embs = torch.cat(anchor_embs, dim=0).numpy() if anchor_embs else np.empty((0, 2))
    positive_embs = torch.cat(positive_embs, dim=0).numpy() if positive_embs else np.empty((0, 2))
    pos_sims_all = torch.cat(pos_sims_all, dim=0).numpy()
    hard_neg_sims_all = torch.cat(hard_neg_sims_all, dim=0).numpy()

    metrics["similarity_gap"] = metrics["positive_similarity_mean"] - metrics["negative_similarity_mean"]
    metrics["num_samples_evaluated"] = int(total_weight)

    artifacts = {
        "anchor_embeddings": anchor_embs,
        "positive_embeddings": positive_embs,
        "positive_similarities": pos_sims_all,
        "hardest_negative_similarities": hard_neg_sims_all,
    }

    return metrics, artifacts


def save_pca_plot(anchor_embs: np.ndarray, positive_embs: np.ndarray, out_path: Path):
    if len(anchor_embs) == 0 or len(positive_embs) == 0:
        return

    X = np.concatenate([anchor_embs, positive_embs], axis=0)
    X = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[:2].T

    n_anchor = len(anchor_embs)
    proj_anchor = proj[:n_anchor]
    proj_positive = proj[n_anchor:]

    plt.figure(figsize=(7, 6))
    plt.scatter(proj_anchor[:, 0], proj_anchor[:, 1], s=10, alpha=0.5, label="anchor")
    plt.scatter(proj_positive[:, 0], proj_positive[:, 1], s=10, alpha=0.5, label="positive")
    plt.title("Embedding PCA")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_similarity_histogram(pos_sims: np.ndarray, hard_neg_sims: np.ndarray, out_path: Path):
    plt.figure(figsize=(7, 5))
    plt.hist(pos_sims, bins=40, alpha=0.6, label="positive similarity")
    plt.hist(hard_neg_sims, bins=40, alpha=0.6, label="hardest negative similarity")
    plt.title("Similarity distributions")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--family", type=str, default="euclidean")
    parser.add_argument("--allow-reflection", action="store_true")

    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)

    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--val-seed", type=int, default=2)
    parser.add_argument("--test-seed", type=int, default=3)

    parser.add_argument("--num-curve-points", type=int, default=300)
    parser.add_argument("--fourier-max-freq", type=int, default=5)
    parser.add_argument("--fourier-scale", type=float, default=0.9)
    parser.add_argument("--fourier-decay-power", type=float, default=2.0)
    parser.add_argument("--curve-max-tries", type=int, default=300)
    parser.add_argument("--curve-min-size", type=float, default=0.45)
    parser.add_argument("--curve-max-size", type=float, default=0.75)

    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--half-width", type=int, default=14)
    parser.add_argument("--num-negatives", type=int, default=8)
    parser.add_argument("--negative-min-offset", type=int, default=6)
    parser.add_argument("--negative-max-offset", type=int, default=30)
    parser.add_argument("--patch-mode", type=str, default="jittered_symmetric")
    parser.add_argument("--jitter-fraction", type=float, default=0.25)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-vis-samples", type=int, default=5000)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tangent_collate_fn,
    )

    model = TangentPatchEmbeddingModel(
        point_mlp_dims=[64, 64, 128],
        embedding_dim=args.embedding_dim,
        head_dropout=0.1,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    loss_fn = TupleInfoNCELoss(
        temperature=args.temperature,
        normalize=True,
        use_in_batch_negatives=True,
    )

    metrics, artifacts = evaluate_loader(
        model=model,
        loss_fn=loss_fn,
        loader=loader,
        device=device,
        max_vis_samples=args.max_vis_samples,
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_pca_plot(
        artifacts["anchor_embeddings"],
        artifacts["positive_embeddings"],
        output_dir / "embedding_pca.png",
    )
    save_similarity_histogram(
        artifacts["positive_similarities"],
        artifacts["hardest_negative_similarities"],
        output_dir / "similarity_hist.png",
    )

    print("Evaluation done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
