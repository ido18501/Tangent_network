from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from training.collate import TangentBatch


@dataclass
class TrainOutput:
    loss: float
    stats: Dict[str, float]


class TangentTrainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        grad_clip_norm=None,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm

        self.model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _move_batch(self, batch: TangentBatch):
        batch.anchor = batch.anchor.to(self.device)
        batch.positive = batch.positive.to(self.device)
        batch.negatives = batch.negatives.to(self.device)
        batch.transform_matrix = batch.transform_matrix.to(self.device)
        return batch

    def _forward_pair(self, batch: TangentBatch):
        """
        Runs the model on anchor and positive patches.

        Expected model output:
            {
                "weights": (B, P),
                "vector":  (B, 2)
            }
        """
        anchor_patch = batch.anchor
        positive_patch = batch.positive

        anchor_out = self.model(anchor_patch)
        positive_out = self.model(positive_patch)

        return anchor_out, positive_out

    def train_step(self, batch):
        self.model.train()
        batch = self._move_batch(batch)

        self.optimizer.zero_grad()

        anchor_out, positive_out = self._forward_pair(batch)

        v_anchor = anchor_out["vector"]
        v_positive = positive_out["vector"]
        w_anchor = anchor_out["weights"]

        B, M, P, C = batch.negatives.shape
        flat_neg = batch.negatives.view(B * M, P, C)
        neg_out = self.model(flat_neg)
        v_neg = neg_out["vector"].view(B, M, 2)

        loss, stats = self.loss_fn(
            v_anchor=v_anchor,
            v_positive=v_positive,
            weights_anchor=w_anchor,
            transform_matrix=batch.transform_matrix,
            v_negatives=v_neg,
            return_stats=True,
        )

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        return TrainOutput(
            loss=float(loss.item()),
            stats=stats,
        )

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        batch = self._move_batch(batch)

        anchor_out, positive_out = self._forward_pair(batch)

        v_anchor = anchor_out["vector"]
        v_positive = positive_out["vector"]
        w_anchor = anchor_out["weights"]

        B, M, P, C = batch.negatives.shape
        flat_neg = batch.negatives.view(B * M, P, C)
        neg_out = self.model(flat_neg)
        v_neg = neg_out["vector"].view(B, M, 2)

        loss, stats = self.loss_fn(
            v_anchor=v_anchor,
            v_positive=v_positive,
            weights_anchor=w_anchor,
            transform_matrix=batch.transform_matrix,
            v_negatives=v_neg,
            return_stats=True,
        )

        return TrainOutput(
            loss=float(loss.item()),
            stats=stats,
        )

    def _run_loader(self, loader, train):
        metrics = {}
        n = 0

        for batch in loader:
            if train:
                out = self.train_step(batch)
            else:
                out = self.eval_step(batch)

            for k, v in out.stats.items():
                metrics[k] = metrics.get(k, 0.0) + v

            n += 1

        for k in metrics:
            metrics[k] /= n

        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        early_stopping_patience=10,
    ):
        best_val = float("inf")
        best_epoch = 0
        patience = 0

        best_model_path = self.checkpoint_dir / "best_model.pt"

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True)
            val_metrics = self._run_loader(val_loader, train=False)

            val_loss = val_metrics["loss"]

            print(f"\nEpoch {epoch}")
            print("train:", train_metrics)
            print("val:  ", val_metrics)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0

                torch.save(self.model.state_dict(), best_model_path)
                print("✓ saved new best model")
            else:
                patience += 1

            if patience >= early_stopping_patience:
                print("Early stopping triggered")
                break

        print("\nBest validation epoch:", best_epoch)

        self.model.load_state_dict(torch.load(best_model_path))
        return best_model_path

    def evaluate(self, loader):
        metrics = self._run_loader(loader, train=False)

        print("\nTest metrics:")
        print(metrics)

        return metrics