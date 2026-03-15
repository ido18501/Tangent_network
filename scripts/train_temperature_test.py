import torch
from torch.utils.data import DataLoader

from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from training.losses import TupleInfoNCELoss
from training.trainer import TangentTrainer

from models.tangent_model import TangentPatchEmbeddingModel


def build_dataset(size, seed):

    return TangentDataset(
        length=size,
        family="euclidean",
        patch_size=9,
        half_width=14,
        num_negatives=8,
        negative_min_offset=6,
        negative_max_offset=30,
        seed=seed,
    )


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # -------------------------
    # Dataset sizes
    # -------------------------

    train_size = 50000
    val_size = 5000
    test_size = 5000

    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = build_dataset(train_size, seed=1)
    val_dataset = build_dataset(val_size, seed=2)
    test_dataset = build_dataset(test_size, seed=3)

    # -------------------------
    # DataLoaders
    # -------------------------

    batch_size = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=tangent_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=tangent_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=tangent_collate_fn,
        pin_memory=True,
    )

    # -------------------------
    # Model
    # -------------------------

    model = TangentPatchEmbeddingModel(
        point_mlp_dims=[64, 64, 128],
        embedding_dim=64,
        head_dropout=0.1,
    )

    # -------------------------
    # Loss
    # -------------------------

    loss_fn = TupleInfoNCELoss(
        temperature=0.07
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # -------------------------
    # Trainer
    # -------------------------

    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        grad_clip_norm=1.0,
        checkpoint_dir="checkpoints_temp_test",
    )

    # -------------------------
    # Train
    # -------------------------

    best_model_path = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=40,
        early_stopping_patience=8,
    )

    # -------------------------
    # Final Test
    # -------------------------

    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
