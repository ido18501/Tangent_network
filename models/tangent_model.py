from __future__ import annotations

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], use_batchnorm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")
        layers = []
        prev_dim = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, P, C), got {tuple(x.shape)}")
        B, P, _ = x.shape
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                out = out.reshape(B * P, -1)
                out = layer(out)
                out = out.reshape(B, P, -1)
            else:
                out = layer(out)
        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TangentOperatorModel(nn.Module):
    """
    Single-head operator model.

    The single head learns a first-derivative-like operator over the patch.
    The second-derivative-like output is obtained by reapplying the same learned
    weights to the already weighted patch, so the second-order loss is preserved
    without a separate second head.

    No output normalization and no zero-sum enforcement are imposed.
    """

    def __init__(
        self,
        patch_size: int,
        point_dim: int = 2,
        point_mlp_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        use_batchnorm: bool = True,
        point_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if point_mlp_dims is None:
            point_mlp_dims = [64, 64, 128]
        if head_dims is None:
            head_dims = [128, 64]

        self.patch_size = patch_size
        self.point_encoder = SharedMLP(
            in_dim=point_dim,
            hidden_dims=point_mlp_dims,
            use_batchnorm=use_batchnorm,
            dropout=point_dropout,
        )
        feature_dim = point_mlp_dims[-1]
        pooled_dim = 2 * feature_dim

        self.operator_head_first = MLPHead(
            in_dim=pooled_dim,
            hidden_dims=head_dims,
            out_dim=patch_size,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (B, P, 2), got {tuple(x.shape)}")

        point_features = self.point_encoder(x)
        mean_feat = point_features.mean(dim=1)
        max_feat = point_features.max(dim=1).values
        patch_feature = torch.cat([mean_feat, max_feat], dim=-1)

        weights = self.operator_head_first(patch_feature)

        v_first = torch.einsum("bp,bpd->bd", weights, x)

        weighted_patch = weights.unsqueeze(-1) * x
        v_second = torch.einsum("bp,bpd->bd", weights, weighted_patch)

        return {
            "weights_first": weights,
            "weights_second": weights,
            "weights": weights,
            "vector_first": v_first,
            "vector_second": v_second,
            "vector": v_first,
        }
