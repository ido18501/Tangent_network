from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tangent_model import TangentOperatorModel
from losses import OperatorEuclideanDerivativeLoss
from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from training.trainer import TangentTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=9)
    p.add_argument("--half-width", type=int, default=12)
    p.add_argument("--num-curve-points", type=int, default=1000)
    p.add_argument("--num-negatives", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/test_run")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def make_dataset(length: int, args) -> TangentDataset:
    return TangentDataset(
        length=length,
        family="euclidean",
        num_curve_points=args.num_curve_points,
        patch_size=args.patch_size,
        half_width=args.half_width,
        num_negatives=args.num_negatives,
        negative_min_offset=5,
        negative_max_offset=25,
        patch_mode="random_warp_symmetric",
        jitter_fraction=0.25,
        closed=True,
        return_centered=True,
        gt_dense_num_points=4096,
        seed=args.seed,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Using device: {args.device}", flush=True)

    train_dataset = make_dataset(args.train_size, args)
    val_dataset = make_dataset(args.val_size, args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        collate_fn=tangent_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        collate_fn=tangent_collate_fn,
    )

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        point_dim=2,
        point_mlp_dims=[64, 64, 128],
        head_dims=[256, 128],
        use_batchnorm=True,
        point_dropout=0.0,
        head_dropout=0.0,
        operator_row_rms_target=1.0,
    )

    loss_fn = OperatorEuclideanDerivativeLoss(
        lambda_reg=1e-4,
        lambda_neg=0.1,
        neg_margin=0.05,
        lambda_first=1.0,
        lambda_second=1.0,
        lambda_equiv_first=1.0,
        lambda_equiv_second=1.0,
        lambda_first_norm=0.05,
        second_scale_floor=0.05,
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
        device=args.device,
        grad_clip_norm=1.0,
        checkpoint_dir=args.checkpoint_dir,
    )

    best_model_path = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=max(2, args.epochs),
    )

    print(f"\nBest model saved to: {best_model_path}", flush=True)


if __name__ == "__main__":
    main()
