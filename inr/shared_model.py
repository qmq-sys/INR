"""Shared Spatial INR + subject-specific latent code.

Architecture change only:
  Independent: D_s(x) = f_{θ_s}(PE(x))
  Shared:      D_s(x) = f_θ(PE(x), z_s)

DTI parameterization (Cholesky → D), activations, PE, hidden/layers
match SpatialDTIINR in ``inr.model`` — that class is not modified.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import FourierFeatures


def _params_to_s0_d(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same Cholesky decode as SpatialDTIINR.forward (frozen parameterization)."""
    logS0 = params[:, 0:1].clamp(-5.0, 12.0)
    L_raw = params[:, 1:]
    S0 = torch.exp(logS0).squeeze(-1)

    L11 = torch.exp(L_raw[:, 0].clamp(-8.0, -1.5))
    L22 = torch.exp(L_raw[:, 1].clamp(-8.0, -1.5))
    L33 = torch.exp(L_raw[:, 2].clamp(-8.0, -1.5))
    L21 = L_raw[:, 3].tanh() * 0.05
    L31 = L_raw[:, 4].tanh() * 0.05
    L32 = L_raw[:, 5].tanh() * 0.05
    zeros = torch.zeros_like(L11)
    row1 = torch.stack([L11, zeros, zeros], dim=-1)
    row2 = torch.stack([L21, L22, zeros], dim=-1)
    row3 = torch.stack([L31, L32, L33], dim=-1)
    Lmat = torch.stack([row1, row2, row3], dim=-2)
    D = Lmat @ Lmat.transpose(-1, -2)
    return S0, D


def _build_mlp_head(in_dim: int, hidden: int, layers: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    last = in_dim
    for _ in range(int(layers)):
        blocks.append(nn.Linear(last, hidden))
        blocks.append(nn.ReLU(inplace=True))
        last = hidden
    blocks.append(nn.Linear(last, 7))
    mlp = nn.Sequential(*blocks)
    nn.init.zeros_(mlp[-1].weight)
    with torch.no_grad():
        mlp[-1].bias.copy_(
            torch.tensor(
                [6.0, -3.45, -3.45, -3.45, 0.0, 0.0, 0.0],
                dtype=torch.float32,
            )
        )
    return mlp


class SharedSpatialDTIINR(nn.Module):
    """
    Shared INR with learnable subject embedding:
      h = concat(PE(x), z_s)
      (S0, D) = decode(f_θ(h))
    """

    def __init__(
        self,
        num_subjects: int,
        *,
        latent_dim: int = 32,
        hidden: int = 128,
        layers: int = 4,
        pe_freqs: int = 8,
    ):
        super().__init__()
        self.num_subjects = int(num_subjects)
        self.latent_dim = int(latent_dim)
        self.pe = FourierFeatures(n_freqs=pe_freqs, include_input=True)
        self.subject_embedding = nn.Embedding(self.num_subjects, self.latent_dim)
        nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.01)
        in_dim = self.pe.out_dim + self.latent_dim
        self.mlp = _build_mlp_head(in_dim, hidden=int(hidden), layers=int(layers))

    def forward(
        self,
        xyz_m11: torch.Tensor,
        subject_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz_m11: [V, 3] normalized coordinates in [-1, 1]
            subject_idx: [V] long indices, or scalar / [1] broadcast to all voxels
        """
        pe_feat = self.pe(xyz_m11)
        idx = subject_idx.reshape(-1).long()
        if idx.numel() == 1:
            z = self.subject_embedding(idx).expand(pe_feat.shape[0], -1)
        elif idx.numel() == pe_feat.shape[0]:
            z = self.subject_embedding(idx)
        else:
            raise ValueError(
                f"subject_idx size {idx.numel()} must be 1 or match voxels {pe_feat.shape[0]}"
            )
        params = self.mlp(torch.cat([pe_feat, z], dim=-1))
        return _params_to_s0_d(params)
