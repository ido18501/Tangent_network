from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentPatchEmbeddingModel
from training.collate import tangent_collate_fn
from training.losses import TupleInfoNCELoss
from training.trainer import TangentTrainer
from scripts.evaluate_model import evaluate_loader


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x]


def parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x]


def make_dataset(length, seed, family, patch_size, half_width, num_negatives, neg_min, neg_max, allow_reflection):
    return TangentDataset(
        length=length,
        family=family,
        num_curve_points=300,
        fourier_max_freq=5,
        fourier_scale=0.9,
        fourier_decay_power=2.0,
        curve_max_tries=300,
        curve_min_size=0.45,
        curve_max_size=0.75,
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=neg_min,
        negative_max_offset=neg_max,
        patch_mode="jittered_symmetric",
        jitter_fraction=0.25,
        closed=True,
        transform_kwargs={"allow_reflection": allow_reflection},
        return_centered=True,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--family", type=str, default="euclidean")
    parser.add_argument("--allow-reflection", action="store_true")

    parser.add_argument("--patch-sizes", type=str, default="9,11")
    parser.add_argument("--embedding-dims", type=str, default="64,128")
    parser.add_argument("--temperatures", type=str, default="0.05,0.07,0.1")
    parser.add_argument("--negative-min-offsets", type=str, default="6,15")
    parser.add_argument("--negative-max-offsets", type=str, default="30,40")

    parser.add_argument("--half-width", type=int, default=14)
    parser.add_argument("--num-negatives", type=int, default=8)

    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=3000)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    patch_sizes = parse_int_list(args.patch_sizes)
    embedding_dims = parse_int_list(args.embedding_dims)
    temperatures = parse_float_list(args.temperatures)
    neg_mins = parse_int_list(args.negative_min_offsets)
    neg_maxs = parse_int_list(args.negative_max_offsets)

    summary = []

    run_idx = 0
    for patch_size, embedding_dim, temperature, neg_min, neg_max in itertools.product(
        patch_sizes, embedding_dims, temperatures, neg_mins, neg_maxs
    ):
        if neg_max < neg_min:
            continue

        run_idx += 1
        run_name = f"run_{run_idx:03d}_fam-{args.family}_patch-{patch_size}_emb-{embedding_dim}_temp-{temperature}_neg-{neg_min}-{neg_max}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "family": args.family,
            "allow_reflection": args.allow_reflection,
            "patch_size": patch_size,
            "embedding_dim": embedding_dim,
            "temperature": temperature,
            "negative_min_offset": neg_min,
            "negative_max_offset": neg_max,
            "half_width": args.half_width,
            "num_negatives": args.num_negatives,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n===== {run_name} =====")

        train_ds = make_dataset(args.train_size, 1, args.family, patch_size, args.half_width, args.num_negatives, neg_min, neg_max, args.allow_reflection)
        val_ds = make_dataset(args.val_size, 2, args.family, patch_size, args.half_width, args.num_negatives, neg_min, neg_max, args.allow_reflection)
        test_ds = make_dataset(args.test_size, 3, args.family, patch_size, args.half_width, args.num_negatives, neg_min, neg_max, args.allow_reflection)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=tangent_collate_fn)

        model = TangentPatchEmbeddingModel(
            point_mlp_dims=[64, 64, 128],
            embedding_dim=embedding_dim,
            head_dropout=0.1,
        )

        loss_fn = TupleInfoNCELoss(
            temperature=temperature,
            normalize=True,
            use_in_batch_negatives=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        trainer = TangentTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=1.0,
            checkpoint_dir=run_dir / "checkpoints",
        )

        best_model_path = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.patience,
        )

        # Reload best model explicitly
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        val_metrics, _ = evaluate_loader(model, loss_fn, val_loader, device, max_vis_samples=0)
        test_metrics, _ = evaluate_loader(model, loss_fn, test_loader, device, max_vis_samples=0)

        with open(run_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)
        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        record = {
            "run_name": run_name,
            **config,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        summary.append(record)

        with open(output_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\nGrid search finished.")
    print(f"Saved summary to: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
