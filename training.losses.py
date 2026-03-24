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
        lambda_second_dir_max: float = 0.05,
        lambda_second_mag_max: float = 0.005,
        second_warmup_epochs: int = 3,
        second_ramp_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_neg = lambda_neg
        self.neg_margin = neg_margin
        self.lambda_first = lambda_first
        self.lambda_second = lambda_second
        self.lambda_equiv_first = lambda_equiv_first
        self.lambda_equiv_second = lambda_equiv_second
        self.lambda_second_dir_max = lambda_second_dir_max
        self.lambda_second_mag_max = lambda_second_mag_max
        self.second_warmup_epochs = second_warmup_epochs
        self.second_ramp_epochs = second_ramp_epochs

    def _ramp_weight(self, current_epoch: int, max_value: float) -> float:
        if current_epoch < self.second_warmup_epochs:
            return 0.0
        if self.second_ramp_epochs <= 0:
            return max_value
        progress = (current_epoch - self.second_warmup_epochs + 1) / float(self.second_ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        return max_value * progress

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
        current_epoch: int = 0,
    ):
        target_first = torch.einsum("bij,bj->bi", transform_matrix, v_first_anchor)
        target_second = torch.einsum("bij,bj->bi", transform_matrix, v_second_anchor)

        equiv_first_loss = F.mse_loss(v_first_positive, target_first)
        equiv_second_loss = F.mse_loss(v_second_positive, target_second)
        first_loss = F.mse_loss(v_first_anchor, gt_first_anchor)
        eps = 1e-8

        pred_second_norm = v_second_anchor.norm(dim=-1)
        gt_second_norm = gt_second_anchor.norm(dim=-1)

        pred_second_unit = v_second_anchor / (pred_second_norm.unsqueeze(-1) + eps)
        gt_second_unit = gt_second_anchor / (gt_second_norm.unsqueeze(-1) + eps)

        second_cosine = (pred_second_unit * gt_second_unit).sum(dim=-1).clamp(-1.0, 1.0)
        second_dir_loss = (1.0 - second_cosine).mean()

        second_mag_loss = F.smooth_l1_loss(pred_second_norm, gt_second_norm)

        reg_loss = weights_first_anchor.pow(2).mean()

        neg_loss = torch.tensor(0.0, device=v_first_anchor.device, dtype=v_first_anchor.dtype)
        if v_first_negatives is not None and self.lambda_neg > 0.0:
            diff_neg = v_first_negatives - target_first.unsqueeze(1)
            neg_dist_sq = (diff_neg ** 2).sum(dim=-1)
            neg_loss = F.relu(self.neg_margin - neg_dist_sq).mean()

        lambda_second_dir = self._ramp_weight(current_epoch, self.lambda_second_dir_max)
        lambda_second_mag = self._ramp_weight(current_epoch, self.lambda_second_mag_max)
        loss = (
                self.lambda_equiv_first * equiv_first_loss
                + self.lambda_equiv_second * equiv_second_loss
                + self.lambda_first * first_loss
                + lambda_second_dir * second_dir_loss
                + lambda_second_mag * second_mag_loss
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
                "second_loss": float((lambda_second_dir * second_dir_loss + lambda_second_mag * second_mag_loss).detach().item()),
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
                "second_dir_loss": float(second_dir_loss.detach().item()),
                "second_mag_loss": float(second_mag_loss.detach().item()),
                "second_cosine_mean": float(second_cosine.mean().detach().item()),
                "second_mag_ratio_mean": float((pred_second_norm / (gt_second_norm + eps)).mean().detach().item()),
                "lambda_second_dir": float(lambda_second_dir),
                "lambda_second_mag": float(lambda_second_mag),
            }
        return loss, stats
