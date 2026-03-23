from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from training.collate import TangentBatch


@dataclass
class TrainOutput:
    loss: float
    stats: Dict[str, float]


class TangentTrainer:
    def __init__(self, model, optimizer, loss_fn, device, grad_clip_norm=None, checkpoint_dir="checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _move_batch(self, batch: TangentBatch) -> TangentBatch:
        batch.anchor = batch.anchor.to(self.device)
        batch.positive = batch.positive.to(self.device)
        batch.negatives = batch.negatives.to(self.device)
        batch.transform_matrix = batch.transform_matrix.to(self.device)
        batch.gt_first_anchor = batch.gt_first_anchor.to(self.device)
        batch.gt_second_anchor = batch.gt_second_anchor.to(self.device)
        return batch

    def _forward_pair(self, batch: TangentBatch):
        return self.model(batch.anchor), self.model(batch.positive)

    def train_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.train()
        batch = self._move_batch(batch)
        self.optimizer.zero_grad()

        anchor_out, positive_out = self._forward_pair(batch)

        B, M, P, C = batch.negatives.shape
        flat_neg = batch.negatives.view(B * M, P, C)
        neg_out = self.model(flat_neg)
        v_first_neg = neg_out["vector_first"].view(B, M, 2)

        loss, stats = self.loss_fn(
            v_first_anchor=anchor_out["vector_first"],
            v_first_positive=positive_out["vector_first"],
            v_second_anchor=anchor_out["vector_second"],
            v_second_positive=positive_out["vector_second"],
            weights_first_anchor=anchor_out["weights_first"],
            weights_second_anchor=anchor_out["weights_second"],
            transform_matrix=batch.transform_matrix,
            gt_first_anchor=batch.gt_first_anchor,
            gt_second_anchor=batch.gt_second_anchor,
            v_first_negatives=v_first_neg,
            return_stats=True,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return TrainOutput(loss=float(loss.item()), stats=stats)

    @torch.no_grad()
    def eval_step(self, batch: TangentBatch) -> TrainOutput:
        self.model.eval()
        batch = self._move_batch(batch)
        anchor_out, positive_out = self._forward_pair(batch)

        B, M, P, C = batch.negatives.shape
        flat_neg = batch.negatives.view(B * M, P, C)
        neg_out = self.model(flat_neg)
        v_first_neg = neg_out["vector_first"].view(B, M, 2)

        loss, stats = self.loss_fn(
            v_first_anchor=anchor_out["vector_first"],
            v_first_positive=positive_out["vector_first"],
            v_second_anchor=anchor_out["vector_second"],
            v_second_positive=positive_out["vector_second"],
            weights_first_anchor=anchor_out["weights_first"],
            weights_second_anchor=anchor_out["weights_second"],
            transform_matrix=batch.transform_matrix,
            gt_first_anchor=batch.gt_first_anchor,
            gt_second_anchor=batch.gt_second_anchor,
            v_first_negatives=v_first_neg,
            return_stats=True,
        )
        return TrainOutput(loss=float(loss.item()), stats=stats)

    def _run_loader(self, loader, train: bool):
        metrics = {}
        n = 0
        for batch in loader:
            out = self.train_step(batch) if train else self.eval_step(batch)
            for k, v in out.stats.items():
                metrics[k] = metrics.get(k, 0.0) + v
            n += 1
        for k in metrics:
            metrics[k] /= max(n, 1)
        return metrics

    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        best_val = float("inf")
        best_epoch = 0
        patience = 0
        best_model_path = self.checkpoint_dir / "best_model.pt"
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_loader(train_loader, train=True)
            val_metrics = self._run_loader(val_loader, train=False)
            val_loss = val_metrics["loss"]
            print(f"\nEpoch {epoch}", flush=True)
            print("train:", train_metrics, flush=True)
            print("val:  ", val_metrics, flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience = 0
                torch.save(self.model.state_dict(), best_model_path)
                print("✓ saved new best model", flush=True)
            else:
                patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping triggered", flush=True)
                break
        print("\nBest validation epoch:", best_epoch, flush=True)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return best_model_path

    def evaluate(self, loader):
        metrics = self._run_loader(loader, train=False)
        print("\nTest metrics:", flush=True)
        print(metrics, flush=True)
        return metrics
