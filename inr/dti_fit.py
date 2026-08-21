"""Traditional WLS-DTI fitting (DIPY) and map extraction."""
from __future__ import annotations

from typing import Any

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity

from .hcp_io import normalize_bvecs, shell_volume_mask


def tensor_to_lower6(D: np.ndarray) -> dict[str, np.ndarray]:
    """D [...,3,3] -> Dxx,Dyy,Dzz,Dxy,Dxz,Dyz."""
    return {
        "Dxx": D[..., 0, 0].astype(np.float32),
        "Dyy": D[..., 1, 1].astype(np.float32),
        "Dzz": D[..., 2, 2].astype(np.float32),
        "Dxy": D[..., 0, 1].astype(np.float32),
        "Dxz": D[..., 0, 2].astype(np.float32),
        "Dyz": D[..., 1, 2].astype(np.float32),
    }


def fit_wls_dti(
    data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    mask: np.ndarray,
    *,
    b0_threshold: float = 50.0,
) -> dict[str, Any]:
    data = np.asarray(data, dtype=np.float64)
    bvals = np.asarray(bvals, dtype=np.float64).ravel()
    bvecs = np.asarray(bvecs, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(mask, dtype=bool)
    if data.ndim != 4:
        raise ValueError(f"data must be 4D, got {data.shape}")

    bvecs_n = normalize_bvecs(bvals, bvecs, b0_threshold=b0_threshold)
    gtab = gradient_table(bvals, bvecs=bvecs_n, b0_threshold=float(b0_threshold))
    tenfit = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(data, mask=mask)

    D = np.asarray(tenfit.quadratic_form, dtype=np.float64)
    D = 0.5 * (D + np.swapaxes(D, -1, -2))

    if getattr(tenfit, "S0_hat", None) is not None:
        S0_raw = np.asarray(tenfit.S0_hat, dtype=np.float64)
    else:
        b0 = bvals < float(b0_threshold)
        S0_raw = np.mean(data[..., b0], axis=-1)

    s0_ok = np.isfinite(S0_raw) & (S0_raw > 0.0) & (S0_raw < 1.0e6)
    S0 = np.where(s0_ok, S0_raw, 0.0).astype(np.float32)

    evals = np.asarray(tenfit.evals, dtype=np.float64)
    evecs = np.asarray(tenfit.evecs, dtype=np.float64)
    order = np.argsort(-evals, axis=-1)
    evals = np.take_along_axis(evals, order, axis=-1)
    evecs = np.take_along_axis(evecs, order[..., None, :], axis=-1)

    fa_raw = fractional_anisotropy(evals)
    md_raw = mean_diffusivity(evals)
    ad_raw = evals[..., 0]
    rd_raw = 0.5 * (evals[..., 1] + evals[..., 2])
    v1_raw = evecs[..., :, 0]

    fa = np.nan_to_num(fa_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    md = np.nan_to_num(md_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ad = np.nan_to_num(ad_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rd = np.nan_to_num(rd_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    v1 = np.nan_to_num(v1_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    nrm = np.linalg.norm(v1, axis=-1, keepdims=True)
    v1 = np.divide(v1, np.maximum(nrm, 1e-12), dtype=np.float32)

    d_ok = np.all(np.isfinite(D), axis=(-1, -2))
    valid = (
        mask
        & s0_ok
        & d_ok
        & np.isfinite(fa_raw)
        & np.isfinite(md_raw)
        & np.isfinite(ad_raw)
        & np.isfinite(rd_raw)
        & np.all(np.isfinite(v1_raw), axis=-1)
        & (evals[..., 2] > 0)
    )

    comps = tensor_to_lower6(D)
    return {
        "S0": S0,
        "D": D.astype(np.float32),
        "FA": fa,
        "MD": md,
        "AD": ad,
        "RD": rd,
        "V1": v1,
        "evals": evals.astype(np.float32),
        "valid_mask": valid.astype(bool),
        **comps,
    }


def fit_dti_b0_b1000(
    data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    brain_mask: np.ndarray,
    *,
    b0_threshold: float = 50.0,
    shell_tol: float = 200.0,
) -> dict[str, Any]:
    """Traditional DTI baseline on b0 ∪ b≈1000."""
    m = shell_volume_mask(
        bvals,
        b0_threshold=b0_threshold,
        shell_tol=shell_tol,
        shells=(1000.0,),
        include_b0=True,
    )
    out = fit_wls_dti(data[..., m], bvals[m], bvecs[m], brain_mask, b0_threshold=b0_threshold)
    out["used_volume_mask"] = m
    out["n_volumes_used"] = int(np.count_nonzero(m))
    return out
