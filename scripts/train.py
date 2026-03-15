from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets.tangent_dataset import TangentDataset
from training.collate import tangent_collate_fn
from training.losses import TupleInfoNCELoss
from training.trainer import TangentTrainer


class DummyPatchEncoder(nn.Module):

    def __init__(self, patch_size, embedding_dim=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):

        return self.net(x)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = 9
    half_width = 12
    num_negatives = 6

    batch_size = 32
    num_epochs = 50

    train_dataset = TangentDataset(
        length=2000,
        family="euclidean",
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=5,
        negative_max_offset=25,
        seed=1,
    )

    val_dataset = TangentDataset(
        length=400,
        family="euclidean",
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=5,
        negative_max_offset=25,
        seed=2,
    )

    test_dataset = TangentDataset(
        length=400,
        family="euclidean",
        patch_size=patch_size,
        half_width=half_width,
        num_negatives=num_negatives,
        negative_min_offset=5,
        negative_max_offset=25,
        seed=3,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tangent_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tangent_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tangent_collate_fn,
    )

    model = DummyPatchEncoder(patch_size)

    loss_fn = TupleInfoNCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = TangentTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        grad_clip_norm=1.0,
    )

    best_model_path = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
    )

    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
