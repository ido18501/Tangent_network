from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OperatorEuclideanDerivativeLoss(nn.Module):
    def __init__(
        self,
        lambda_reg: float = 1e-4,
        lambda_neg: float = 0.1,
        neg_margin: float = 0.05,
        lambda_first: float = 1.0,
        lambda_second: float = 1.0,
        lambda_equiv_first: float = 1.0,
        lambda_equiv_second: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_neg = lambda_neg
        self.neg_margin = neg_margin
        self.lambda_first = lambda_first
        self.lambda_second = lambda_second
        self.lambda_equiv_first = lambda_equiv_first
        self.lambda_equiv_second = lambda_equiv_second

    def forward(
        self,
        *,
        v_first_anchor: torch.Tensor,
        v_first_positive: torch.Tensor,
        v_second_anchor: torch.Tensor,
        v_second_positive: torch.Tensor,
        weights_first_anchor: torch.Tensor,
        weights_second_anchor: torch.Tensor,
        transform_matrix: torch.Tensor,
        gt_first_anchor: torch.Tensor,
        gt_second_anchor: torch.Tensor,
        v_first_negatives: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        target_first = torch.einsum("bij,bj->bi", transform_matrix, v_first_anchor)
        target_second = torch.einsum("bij,bj->bi", transform_matrix, v_second_anchor)

        equiv_first_loss = F.mse_loss(v_first_positive, target_first)
        equiv_second_loss = F.mse_loss(v_second_positive, target_second)
        first_loss = F.mse_loss(v_first_anchor, gt_first_anchor)
        second_loss = F.mse_loss(v_second_anchor, gt_second_anchor)

        reg_loss = weights_first_anchor.pow(2).mean()

        neg_loss = torch.tensor(0.0, device=v_first_anchor.device, dtype=v_first_anchor.dtype)
        if v_first_negatives is not None and self.lambda_neg > 0.0:
            diff_neg = v_first_negatives - target_first.unsqueeze(1)
            neg_dist_sq = (diff_neg ** 2).sum(dim=-1)
            neg_loss = F.relu(self.neg_margin - neg_dist_sq).mean()

        loss = (
            self.lambda_equiv_first * equiv_first_loss
            + self.lambda_equiv_second * equiv_second_loss
            + self.lambda_first * first_loss
            + self.lambda_second * second_loss
            + self.lambda_neg * neg_loss
            + self.lambda_reg * reg_loss
        )

        if not return_stats:
            return loss

        with torch.no_grad():
            stats = {
                "loss": float(loss.detach().item()),
                "equiv_first_loss": float(equiv_first_loss.detach().item()),
                "equiv_second_loss": float(equiv_second_loss.detach().item()),
                "first_loss": float(first_loss.detach().item()),
                "second_loss": float(second_loss.detach().item()),
                "neg_loss": float(neg_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
                "first_vector_norm_mean": float(v_first_anchor.norm(dim=-1).mean().detach().item()),
                "second_vector_norm_mean": float(v_second_anchor.norm(dim=-1).mean().detach().item()),
                "gt_first_norm_mean": float(gt_first_anchor.norm(dim=-1).mean().detach().item()),
                "gt_second_norm_mean": float(gt_second_anchor.norm(dim=-1).mean().detach().item()),
                "first_error_norm_mean": float((v_first_anchor - gt_first_anchor).norm(dim=-1).mean().detach().item()),
                "second_error_norm_mean": float((v_second_anchor - gt_second_anchor).norm(dim=-1).mean().detach().item()),
                "weight_first_sum_mean": float(weights_first_anchor.sum(dim=-1).mean().detach().item()),
                "weight_second_sum_mean": float(weights_second_anchor.sum(dim=-1).mean().detach().item()),
            }
        return loss, stats
