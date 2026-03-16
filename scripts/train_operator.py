from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from training.collate import tangent_collate_fn
from training.losses import OperatorEquivarianceLoss
from training.trainer import TangentTrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = 11
    num_negatives = 8

    train_dataset = TangentDataset(
        length=4000,
        family="euclidean",
        num_curve_points=300,
        fourier_max_freq=5,
        fourier_scale=0.9,
        fourier_decay_power=2.0,
        curve_min_size=0.45,
        curve_max_size=0.75,
        patch_size=patch_size,
        half_width=12,
        half_width_range=(10, 16),
        num_negatives=num_negatives,
        negative_min_offset=8,
        negative_max_offset=40,
        negative_other_curve_fraction=0.5,
        patch_mode="jittered_symmetric",
        jitter_fraction=0.35,
        closed=True,
        transform_kwargs={
            "angle_range": (-3.1415926535, 3.1415926535),
            "allow_reflection": False,
            "translation_range": (-0.25, 0.25),
        },
        return_centered=True,
        point_noise_std=0.003,
        seed=1,
    )

    val_dataset = TangentDataset(
        length=800,
        family="euclidean",
        num_curve_points=300,
        fourier_max_freq=5,
        fourier_scale=0.9,
        fourier_decay_power=2.0,
        curve_min_size=0.45,
        curve_max_size=0.75,
        patch_size=patch_size,
        half_width=12,
        half_width_range=(10, 16),
        num_negatives=num_negatives,
        negative_min_offset=8,
        negative_max_offset=40,
        negative_other_curve_fraction=0.5,
        patch_mode="jittered_symmetric",
        jitter_fraction=0.35,
        closed=True,
        transform_kwargs={
            "angle_range": (-3.1415926535, 3.1415926535),
            "allow_reflection": False,
            "translation_range": (-0.25, 0.25),
        },
        return_centered=True,
        point_noise_std=0.003,
        seed=2,
    )

    test_dataset = TangentDataset(
        length=800,
        family="euclidean",
        num_curve_points=300,
        fourier_max_freq=5,
        fourier_scale=0.9,
        fourier_decay_power=2.0,
        curve_min_size=0.45,
        curve_max_size=0.75,
        patch_size=patch_size,
        half_width=12,
        half_width_range=(10, 16),
        num_negatives=num_negatives,
        negative_min_offset=8,
        negative_max_offset=40,
        negative_other_curve_fraction=0.5,
        patch_mode="jittered_symmetric",
        jitter_fraction=0.35,
        closed=True,
        transform_kwargs={
            "angle_range": (-3.1415926535, 3.1415926535),
            "allow_reflection": False,
            "translation_range": (-0.25, 0.25),
        },
        return_centered=True,
        point_noise_std=0.003,
        seed=3,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=tangent_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=tangent_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=tangent_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = TangentOperatorModel(
        patch_size=patch_size,
        point_dim=2,
        point_mlp_dims=[64, 64, 128],
        head_dims=[128, 64],
        use_batchnorm=True,
        point_dropout=0.0,
        head_dropout=0.0,
    )

    loss_fn = OperatorEquivarianceLoss(
        lambda_reg=1e-4,
        lambda_neg=0.35,
        neg_margin=0.25,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        grad_clip_norm=1.0,
        checkpoint_dir="checkpoints_operator",
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=40,
        early_stopping_patience=8,
    )

    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()