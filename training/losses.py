from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_2d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (B, D), got {tuple(x.shape)}.")


def _check_3d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape (B, M, D), got {tuple(x.shape)}.")


def _normalize_embeddings(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


@dataclass
class InfoNCEStats:
    loss: float
    positive_similarity_mean: float
    negative_similarity_mean: float
    logits_mean: float


class TupleInfoNCELoss(nn.Module):
    """
    InfoNCE loss for tuple data of the form:

        anchor    : (B, D)
        positive  : (B, D)
        negatives : (B, M, D)

    where:
        B = batch size
        M = number of explicit negatives per anchor
        D = embedding dimension

    For each anchor i:
        positive logit  = sim(a_i, p_i) / tau
        negative logits = sim(a_i, n_i,j) / tau

    Optional in-batch negatives:
        all other positives p_k, k != i, are also treated as negatives.

    This is a very good default for your current pipeline.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        use_in_batch_negatives: bool = True,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.temperature = temperature
        self.normalize = normalize
        self.use_in_batch_negatives = use_in_batch_negatives

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        return_stats: bool = False,
    ):
        """
        Args:
            anchor:
                Tensor of shape (B, D)
            positive:
                Tensor of shape (B, D)
            negatives:
                Tensor of shape (B, M, D)
            return_stats:
                If True, also returns a small stats object.

        Returns:
            loss
            or (loss, stats)
        """
        _check_2d("anchor", anchor)
        _check_2d("positive", positive)
        _check_3d("negatives", negatives)

        if anchor.shape != positive.shape:
            raise ValueError(
                f"anchor and positive must have same shape, got "
                f"{tuple(anchor.shape)} and {tuple(positive.shape)}."
            )

        if anchor.shape[0] != negatives.shape[0]:
            raise ValueError(
                "Batch size mismatch between anchor and negatives: "
                f"{anchor.shape[0]} vs {negatives.shape[0]}."
            )

        if anchor.shape[1] != negatives.shape[2]:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"anchor has D={anchor.shape[1]}, negatives have D={negatives.shape[2]}."
            )

        if self.normalize:
            anchor = _normalize_embeddings(anchor)
            positive = _normalize_embeddings(positive)
            negatives = _normalize_embeddings(negatives)

        # Positive logits: (B, 1)
        pos_logits = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        # Explicit negative logits: (B, M)
        neg_logits = torch.einsum("bd,bmd->bm", anchor, negatives) / self.temperature

        logits_parts = [pos_logits, neg_logits]

        # Optional in-batch negatives from other positives
        if self.use_in_batch_negatives:
            batch_logits = torch.matmul(anchor, positive.t()) / self.temperature  # (B, B)

            # Remove diagonal because p_i is the positive for a_i and already included.
            B = anchor.shape[0]
            mask = ~torch.eye(B, dtype=torch.bool, device=batch_logits.device)
            in_batch_neg_logits = batch_logits[mask].view(B, B - 1)  # (B, B-1)

            logits_parts.append(in_batch_neg_logits)

        logits = torch.cat(logits_parts, dim=1)  # (B, 1 + M [+ B-1])

        # Positive is always at class index 0
        targets = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, targets)

        if not return_stats:
            return loss

        with torch.no_grad():
            stats = InfoNCEStats(
                loss=float(loss.detach().item()),
                positive_similarity_mean=float(torch.sum(anchor * positive, dim=-1).mean().item()),
                negative_similarity_mean=float(
                    torch.einsum("bd,bmd->bm", anchor, negatives).mean().item()
                ),
                logits_mean=float(logits.mean().item()),
            )

        return loss, stats


class TripletMarginLossWithHardNegatives(nn.Module):
    """
    Simple triplet-style baseline using the hardest explicit negative per anchor.

    Inputs:
        anchor    : (B, D)
        positive  : (B, D)
        negatives : (B, M, D)

    This is useful as a baseline, but InfoNCE is usually the better default.
    """

    def __init__(
        self,
        margin: float = 0.2,
        normalize: bool = True,
        distance: str = "cosine",
    ) -> None:
        super().__init__()

        if margin <= 0:
            raise ValueError("margin must be positive.")
        if distance not in {"cosine", "euclidean"}:
            raise ValueError("distance must be either 'cosine' or 'euclidean'.")

        self.margin = margin
        self.normalize = normalize
        self.distance = distance

    def _pair_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            sim = torch.sum(x * y, dim=-1)
            return 1.0 - sim
        return torch.norm(x - y, dim=-1, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        _check_2d("anchor", anchor)
        _check_2d("positive", positive)
        _check_3d("negatives", negatives)

        if anchor.shape != positive.shape:
            raise ValueError("anchor and positive must have the same shape.")
        if anchor.shape[0] != negatives.shape[0]:
            raise ValueError("Batch size mismatch between anchor and negatives.")
        if anchor.shape[1] != negatives.shape[2]:
            raise ValueError("Embedding dimension mismatch.")

        if self.normalize:
            anchor = _normalize_embeddings(anchor)
            positive = _normalize_embeddings(positive)
            negatives = _normalize_embeddings(negatives)

        pos_dist = self._pair_distance(anchor, positive)  # (B,)

        # Distance from each anchor to its negatives: (B, M)
        if self.distance == "cosine":
            neg_sim = torch.einsum("bd,bmd->bm", anchor, negatives)
            neg_dist = 1.0 - neg_sim
        else:
            neg_dist = torch.norm(anchor.unsqueeze(1) - negatives, dim=-1, p=2)

        hardest_neg_dist = torch.min(neg_dist, dim=1).values  # (B,)

        loss = F.relu(pos_dist - hardest_neg_dist + self.margin).mean()
        return loss






class OperatorEquivarianceLoss(nn.Module):

    def __init__(
        self,
        lambda_reg: float = 1e-4,
        lambda_neg: float = 0.5,
        neg_margin: float = 0.3,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_neg = lambda_neg
        self.neg_margin = neg_margin
        self.eps = eps

    def forward(
        self,
        v_anchor: torch.Tensor,
        v_positive: torch.Tensor,
        weights_anchor: torch.Tensor,
        transform_matrix: torch.Tensor,
        v_negatives: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        # positive equivariance term
        target = torch.einsum("bij,bj->bi", transform_matrix, v_anchor)
        target = F.normalize(target, dim=-1, eps=self.eps)
        v_positive = F.normalize(v_positive, dim=-1, eps=self.eps)

        cosine_pos = torch.sum(target * v_positive, dim=-1)
        equiv_loss = 1.0 - cosine_pos.mean()

        # regularize weights
        reg_loss = weights_anchor.pow(2).mean()

        # optional negative term
        neg_loss = torch.tensor(0.0, device=v_anchor.device)
        neg_alignment = torch.tensor(0.0, device=v_anchor.device)

        if v_negatives is not None:
            v_anchor_n = F.normalize(v_anchor, dim=-1, eps=self.eps)
            v_neg_n = F.normalize(v_negatives, dim=-1, eps=self.eps)

            cosine_neg = torch.einsum("bd,bmd->bm", v_anchor_n, v_neg_n)
            neg_alignment = cosine_neg.mean()
            neg_loss = F.relu(cosine_neg - self.neg_margin).mean()

        loss = equiv_loss + self.lambda_reg * reg_loss + self.lambda_neg * neg_loss

        if not return_stats:
            return loss

        stats = {
            "loss": float(loss.detach().item()),
            "equiv_loss": float(equiv_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "neg_loss": float(neg_loss.detach().item()),
            "pos_alignment": float(cosine_pos.mean().detach().item()),
            "neg_alignment": float(neg_alignment.detach().item()),
            "anchor_vector_norm_mean": float(v_anchor.norm(dim=-1).mean().detach().item()),
            "positive_vector_norm_mean": float(v_positive.norm(dim=-1).mean().detach().item()),
        }

        return loss, stats