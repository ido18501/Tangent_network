from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """
    Shared MLP applied independently to each point in a patch.

    Input:
        x: (B, P, C)

    Output:
        y: (B, P, D)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        layers = []
        prev_dim = in_dim

        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))

            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        self.layers = nn.ModuleList(layers)
        self.use_batchnorm = use_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, C)

        Returns:
            (B, P, D)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, P, C), got {tuple(x.shape)}")

        B, P, _ = x.shape
        out = x

        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                # BatchNorm1d expects (N, C), so flatten point dimension
                out = out.reshape(B * P, -1)
                out = layer(out)
                out = out.reshape(B, P, -1)
            else:
                out = layer(out)

        return out


class MLPHead(nn.Module):
    """
    Small MLP head mapping pooled patch feature -> 2D vector.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        layers = []
        prev_dim = in_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TangentPatchEncoder(nn.Module):
    """
    PCPNet / PointNet-style patch encoder for local tangent prediction.

    Input:
        x: (B, P, 2)
            B = batch size
            P = patch size

    Output:
        y: (B, 2)
            unit-norm 2D vector

    Architecture:
        per-point shared MLP
        -> symmetric pooling (mean + max)
        -> MLP head
        -> L2 normalization
    """

    def __init__(
        self,
        point_dim: int = 2,
        point_mlp_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        use_batchnorm: bool = True,
        point_dropout: float = 0.0,
        head_dropout: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if point_mlp_dims is None:
            point_mlp_dims = [64, 64, 128]

        if head_dims is None:
            head_dims = [128, 64]

        self.eps = eps

        self.point_encoder = SharedMLP(
            in_dim=point_dim,
            hidden_dims=point_mlp_dims,
            use_batchnorm=use_batchnorm,
            dropout=point_dropout,
        )

        feature_dim = point_mlp_dims[-1]
        pooled_dim = 2 * feature_dim  # mean + max

        self.head = MLPHead(
            in_dim=pooled_dim,
            hidden_dims=head_dims,
            out_dim=2,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, 2)

        Returns:
            unit_vectors: (B, 2)
        """
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (B, P, 2), got {tuple(x.shape)}")

        point_features = self.point_encoder(x)           # (B, P, F)

        mean_feat = point_features.mean(dim=1)           # (B, F)
        max_feat = point_features.max(dim=1).values      # (B, F)

        patch_feature = torch.cat([mean_feat, max_feat], dim=-1)  # (B, 2F)

        raw_vec = self.head(patch_feature)               # (B, 2)
        unit_vec = F.normalize(raw_vec, p=2, dim=-1, eps=self.eps)

        return unit_vec


class TangentPatchEmbeddingModel(nn.Module):
    """
    Same encoder backbone, but returns an embedding instead of a direct tangent.

    This can be useful if you want to compare:
        direct tangent prediction
    vs
        contrastive embedding learning

    Input:
        x: (B, P, 2)

    Output:
        z: (B, D), normalized embedding
    """

    def __init__(
        self,
        point_dim: int = 2,
        point_mlp_dims: list[int] | None = None,
        embedding_dim: int = 64,
        use_batchnorm: bool = True,
        point_dropout: float = 0.0,
        head_dropout: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if point_mlp_dims is None:
            point_mlp_dims = [64, 64, 128]

        self.eps = eps

        self.point_encoder = SharedMLP(
            in_dim=point_dim,
            hidden_dims=point_mlp_dims,
            use_batchnorm=use_batchnorm,
            dropout=point_dropout,
        )

        feature_dim = point_mlp_dims[-1]
        pooled_dim = 2 * feature_dim

        self.embedding_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected input shape (B, P, 2), got {tuple(x.shape)}")

        point_features = self.point_encoder(x)           # (B, P, F)

        mean_feat = point_features.mean(dim=1)           # (B, F)
        max_feat = point_features.max(dim=1).values      # (B, F)

        patch_feature = torch.cat([mean_feat, max_feat], dim=-1)  # (B, 2F)

        z = self.embedding_head(patch_feature)           # (B, D)
        z = F.normalize(z, p=2, dim=-1, eps=self.eps)

        return z
