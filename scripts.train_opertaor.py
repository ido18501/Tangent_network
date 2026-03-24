from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.losses import OperatorEuclideanDerivativeLoss
from training.trainer import TangentTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-length", type=int, default=5000)
    p.add_argument("--val-length", type=int, default=1500)
    p.add_argument("--test-length", type=int, default=1500)
    p.add_argument("--train-seed", type=int, default=1000)
    p.add_argument("--val-seed", type=int, default=2000)
    p.add_argument("--test-seed", type=int, default=3000)
    p.add_argument("--global-seed", type=int, default=123)
    p.add_argument("--transform-family", type=str, default="euclidean", choices=["euclidean"])
    p.add_argument("--num-curve-points", type=int, default=300)
    p.add_argument("--fourier-max-freq", type=int, default=5)
    p.add_argument("--fourier-scale", type=float, default=0.9)
    p.add_argument("--fourier-decay-power", type=float, default=2.0)
    p.add_argument("--curve-max-tries", type=int, default=300)
    p.add_argument("--curve-min-size", type=float, default=0.45)
    p.add_argument("--curve-max-size", type=float, default=0.75)
    p.add_argument("--mixed-fourier-prob", type=float, default=1.0)
    p.add_argument("--mixed-piecewise-prob", type=float, default=0.0)
    p.add_argument("--patch-size", type=int, default=11)
    p.add_argument("--half-width", type=int, default=12)
    p.add_argument("--num-negatives", type=int, default=8)
    p.add_argument("--negative-min-offset", type=int, default=5)
    p.add_argument("--negative-max-offset", type=int, default=25)
    p.add_argument("--negative-other-curve-fraction", type=float, default=0.5)
    p.add_argument("--sampling-mode", type=str, default="random_warp_symmetric")
    p.add_argument("--jitter-fraction", type=float, default=0.30)
    p.add_argument("--closed", action="store_true", default=True)
    p.add_argument("--no-closed", dest="closed", action="store_false")
    p.add_argument("--return-centered", action="store_true", default=True)
    p.add_argument("--point-noise-std", type=float, default=0.001)
    p.add_argument("--warp-sampling-prob", type=float, default=0.4)
    p.add_argument("--warp-sampling-strength", type=float, default=0.10)
    p.add_argument("--orthogonal-noise-std", type=float, default=0.002)
    p.add_argument("--gt-dense-num-points", type=int, default=4096)
    p.add_argument("--rotation-deg", type=float, default=30.0)
    p.add_argument("--allow-reflection", action="store_true", default=True)
    p.add_argument("--no-reflection", dest="allow_reflection", action="store_false")
    p.add_argument("--translation-range", type=float, default=0.0)
    p.add_argument("--point-mlp-dims", type=int, nargs="+", default=[64, 64, 128])
    p.add_argument("--head-dims", type=int, nargs="+", default=[128, 64])
    p.add_argument("--use-batchnorm", action="store_true", default=True)
    p.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false")
    p.add_argument("--point-dropout", type=float, default=0.0)
    p.add_argument("--head-dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--lambda-reg", type=float, default=1e-4)
    p.add_argument("--lambda-neg", type=float, default=0.1)
    p.add_argument("--neg-margin", type=float, default=0.05)
    p.add_argument("--lambda-first", type=float, default=1.0)
    p.add_argument("--lambda-second", type=float, default=1.0)
    p.add_argument("--lambda-equiv-first", type=float, default=1.0)
    p.add_argument("--lambda-equiv-second", type=float, default=1.0)
    p.add_argument("--lambda-second-dir-max", type=float, default=0.05)
    p.add_argument("--lambda-second-mag-max", type=float, default=0.005)
    p.add_argument("--second-warmup-epochs", type=int, default=3)
    p.add_argument("--second-ramp-epochs", type=int, default=5)
    p.add_argument("--second-phase-lr-scale", type=float, default=0.2)
    p.add_argument("--raw-lambda-reg", type=float, default=1e-4)
    p.add_argument("--raw-lambda-neg", type=float, default=0.1)
    p.add_argument("--neg-margin", type=float, default=0.05)
    p.add_argument("--raw-lambda-first", type=float, default=1.0)
    p.add_argument("--raw-lambda-equiv-first", type=float, default=1.0)
    p.add_argument("--raw-lambda-equiv-second", type=float, default=1.0)
    p.add_argument("--raw-lambda-second-total-max", type=float, default=0.05)
    p.add_argument("--second-dir-fraction", type=float, default=0.9)
    p.add_argument("--second-warmup-epochs", type=int, default=3)
    p.add_argument("--second-ramp-epochs", type=int, default=5)
    p.add_argument("--second-phase-lr-scale", type=float, default=0.2)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_curve_family_probs(args: argparse.Namespace) -> dict[str, float]:
    probs = {"fourier": args.mixed_fourier_prob, "piecewise": args.mixed_piecewise_prob}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def build_transform_kwargs(args: argparse.Namespace) -> dict:
    angle = np.deg2rad(args.rotation_deg)
    return {
        "angle_range": (-angle, angle),
        "allow_reflection": args.allow_reflection,
        "translation_range": (-args.translation_range, args.translation_range),
    }


def build_dataset(args: argparse.Namespace, length: int, seed: int) -> TangentDataset:
    return TangentDataset(
        length=length,
        family=args.transform_family,
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
        negative_other_curve_fraction=args.negative_other_curve_fraction,
        patch_mode=args.sampling_mode,
        jitter_fraction=args.jitter_fraction,
        closed=args.closed,
        transform_kwargs=build_transform_kwargs(args),
        return_centered=args.return_centered,
        point_noise_std=args.point_noise_std,
        curve_family_probs=build_curve_family_probs(args),
        warp_sampling_prob=args.warp_sampling_prob,
        warp_sampling_strength=args.warp_sampling_strength,
        orthogonal_noise_std=args.orthogonal_noise_std,
        gt_dense_num_points=args.gt_dense_num_points,
        dtype=torch.float32,
        seed=seed,
    )


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=tangent_collate_fn,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    args = parse_args()
    set_global_seed(args.global_seed)
    print("TRAIN SCRIPT STARTED", flush=True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    save_json(run_dir / "args.json", vars(args))
    print(json.dumps(vars(args), indent=2), flush=True)

    train_dataset = build_dataset(args, args.train_length, args.train_seed)
    val_dataset = build_dataset(args, args.val_length, args.val_seed)
    test_dataset = build_dataset(args, args.test_length, args.test_seed)
    print(f"train length: {len(train_dataset)}", flush=True)
    print(f"val length: {len(val_dataset)}", flush=True)
    print(f"test length: {len(test_dataset)}", flush=True)

    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, True)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, False)
    test_loader = build_loader(test_dataset, args.batch_size, args.num_workers, False)

    model = TangentOperatorModel(
        patch_size=args.patch_size,
        point_dim=2,
        point_mlp_dims=args.point_mlp_dims,
        head_dims=args.head_dims,
        use_batchnorm=args.use_batchnorm,
        point_dropout=args.point_dropout,
        head_dropout=args.head_dropout,
    )
    loss_fn = OperatorEuclideanDerivativeLoss(
        raw_lambda_reg=args.raw_lambda_reg,
        raw_lambda_neg=args.raw_lambda_neg,
        neg_margin=args.neg_margin,
        raw_lambda_first=args.raw_lambda_first,
        raw_lambda_equiv_first=args.raw_lambda_equiv_first,
        raw_lambda_equiv_second=args.raw_lambda_equiv_second,
        raw_lambda_second_total_max=args.raw_lambda_second_total_max,
        second_dir_fraction=args.second_dir_fraction,
        second_warmup_epochs=args.second_warmup_epochs,
        second_ramp_epochs=args.second_ramp_epochs,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=checkpoints_dir,
        second_warmup_epochs=args.second_warmup_epochs,
        second_phase_lr_scale=args.second_phase_lr_scale,
    )

    best_model_path = trainer.fit(train_loader, val_loader, args.num_epochs, args.early_stopping_patience)
    test_metrics = trainer.evaluate(test_loader)
    save_json(run_dir / "summary.json", {"best_model_path": str(best_model_path), "test_metrics": test_metrics})
    torch.save(model.state_dict(), run_dir / "final_model.pt")


if __name__ == "__main__":
    main()
