"""Spatial INR: (x,y,z) -> S0, D (SPD via Cholesky)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """NeRF-style positional encoding: [x, sin(2^k π x), cos(...)] over 3 axes."""

    def __init__(self, n_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.include_input = bool(include_input)
        freqs = 2.0 ** torch.arange(self.n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        d = 3 * 2 * self.n_freqs
        return d + 3 if self.include_input else d

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: [V,3] in [-1,1]
        xb = xyz.unsqueeze(-1) * self.freqs  # [V,3,F]
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        enc = torch.cat([sin, cos], dim=-1).reshape(xyz.shape[0], -1)
        if self.include_input:
            return torch.cat([xyz, enc], dim=-1)
        return enc


class SpatialDTIINR(nn.Module):
    """
    Experiment-1 model:
      (x,y,z) -> INR -> S0, D
    """

    def __init__(self, hidden: int = 128, layers: int = 4, pe_freqs: int = 8):
        super().__init__()
        self.pe = FourierFeatures(n_freqs=pe_freqs, include_input=True)
        in_dim = self.pe.out_dim
        blocks: list[nn.Module] = []
        last = in_dim
        for _ in range(int(layers)):
            blocks.append(nn.Linear(last, hidden))
            blocks.append(nn.ReLU(inplace=True))
            last = hidden
        blocks.append(nn.Linear(last, 7))  # logS0 + 6 Cholesky params
        self.mlp = nn.Sequential(*blocks)
        # Start near isotropic D ~ 0.001 mm^2/s and moderate S0.
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(
                torch.tensor(
                    [
                        6.0,  # logS0 ~ 400
                        -3.45,  # L11 ~ sqrt(0.001)
                        -3.45,
                        -3.45,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, xyz_m11: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.mlp(self.pe(xyz_m11))
        logS0 = params[:, 0:1].clamp(-5.0, 12.0)
        L_raw = params[:, 1:]
        S0 = torch.exp(logS0).squeeze(-1)

        # Soft-bound Cholesky diagonals to keep D in a plausible diffusion range
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
