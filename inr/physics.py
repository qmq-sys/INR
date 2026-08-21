"""DTI forward model and related torch helpers."""
from __future__ import annotations

import torch


def dti_forward_signal(
    S0: torch.Tensor,
    D: torch.Tensor,
    bvals: torch.Tensor,
    bvecs: torch.Tensor,
    *,
    b_scale: float = 1.0,
) -> torch.Tensor:
    """
    S = S0 * exp(-(b/b_scale) * g^T D g)

    Physical default: bvals in s/mm^2, D in mm^2/s → b_scale=1.
    If D is stored in 10^-3 mm^2/s units, use b_scale=1000.

    S0: [V]
    D:  [V,3,3]
    bvals: [N]
    bvecs: [N,3]
    returns: [V,N]
    """
    g = bvecs.unsqueeze(0)  # [1,N,3]
    Dg = torch.matmul(D.unsqueeze(1), g.unsqueeze(-1))  # [V,N,3,1]
    gDg = torch.matmul(g.unsqueeze(-2), Dg).squeeze(-1).squeeze(-1)  # [V,N]
    b = (bvals / float(b_scale)).unsqueeze(0)  # [1,N]
    return S0.unsqueeze(-1) * torch.exp(-b * gDg)


def compute_fa_md_ad_rd(D: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """D [V,3,3] -> FA, MD, AD, RD [V]. Robust to NaN / non-SPD batches on CUDA."""
    D32 = 0.5 * (D.float() + D.float().transpose(-1, -2))
    D32 = torch.nan_to_num(D32, nan=0.0, posinf=0.0, neginf=0.0)
    # Tiny ridge for numerical eigendecomposition stability
    eye = torch.eye(3, device=D32.device, dtype=D32.dtype).view(1, 3, 3)
    D32 = D32 + float(eps) * eye
    try:
        evals = torch.linalg.eigvalsh(D32)  # ascending
    except Exception:
        evals = torch.linalg.eigvalsh(D32.cpu()).to(D32.device)
    evals = torch.nan_to_num(evals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    l1, l2, l3 = evals[..., 2], evals[..., 1], evals[..., 0]
    md = (l1 + l2 + l3) / 3.0
    ad = l1
    rd = 0.5 * (l2 + l3)
    num = torch.sqrt(((l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2) * 1.5)
    den = torch.sqrt(l1**2 + l2**2 + l3**2).clamp_min(eps)
    fa = (num / den).clamp(0.0, 1.0)
    return fa, md, ad, rd
