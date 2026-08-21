"""Model-independent common-mask DWI evaluation (Independent INR baseline).

Official common mask (matches train_independent evaluation):
  brain_mask AND WLS_valid_mask

Prediction finiteness is NOT part of the common mask (avoids optimistic bias).

WLS and INR MUST share the same sampled voxel indices (G2).
Sampling matches evaluate_dwi_reconstruction: seed + max_voxels on common coords.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .coords import masked_coords_and_indices, voxel_coords_normalized
from .metrics_schema import dwi_reconstruction_metrics
from .physics import dti_forward_signal

DEFAULT_MAX_VOXELS = 131072
DEFAULT_EVAL_SEED = 42
COMMON_MASK_DEF = "brain & WLS_valid"


def build_common_dwi_eval_mask(
    brain_mask: np.ndarray,
    wls_valid_mask: np.ndarray,
    observed_dwi: np.ndarray | None = None,
) -> np.ndarray:
    """Official common mask: brain ∩ WLS_valid (reuse dti_fit valid_mask)."""
    brain = np.asarray(brain_mask, dtype=bool)
    valid = np.asarray(wls_valid_mask, dtype=bool)
    if brain.shape != valid.shape:
        raise ValueError(f"mask mismatch: brain={brain.shape} valid={valid.shape}")
    if observed_dwi is not None:
        obs = np.asarray(observed_dwi)
        if obs.ndim != 4 or obs.shape[:3] != brain.shape:
            raise ValueError(f"observed_dwi spatial mismatch: {getattr(obs, 'shape', None)}")
    return brain & valid


def sample_eval_voxel_indices(
    common_mask: np.ndarray,
    *,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    seed: int = DEFAULT_EVAL_SEED,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Same sampling as evaluate_dwi_reconstruction.

    Returns:
      sampled_flat_idx: [V] flat X*Y*Z indices (shared by WLS and INR)
      sampled_coords:   [V,3] normalized coords
      n_eval_voxels:    full common-mask count before sampling
    """
    coords_all, flat_all = masked_coords_and_indices(common_mask)
    n_eval = int(coords_all.shape[0])
    if n_eval == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
            0,
        )
    max_voxels = int(max_voxels)
    rng = np.random.default_rng(int(seed))
    sel = np.arange(n_eval) if n_eval <= max_voxels else rng.choice(n_eval, size=max_voxels, replace=False)
    return flat_all[sel].astype(np.int64), coords_all[sel].astype(np.float32), n_eval


def get_common_eval_indices(
    common_mask: np.ndarray,
    *,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    seed: int = DEFAULT_EVAL_SEED,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Public alias: identical sampled indices for WLS and INR DWI RelMSE."""
    return sample_eval_voxel_indices(common_mask, max_voxels=max_voxels, seed=seed)


def coords_from_flat_indices(shape_xyz: tuple[int, int, int], flat_idx: np.ndarray) -> np.ndarray:
    """Normalized [-1,1] coords for the given flat voxel indices."""
    all_coords = voxel_coords_normalized(shape_xyz)
    return all_coords[np.asarray(flat_idx, dtype=np.int64)]


def dwi_metrics_no_silent_drop(
    pred: np.ndarray | Any,
    obs: np.ndarray | Any,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    RelMSE / MAE without silently dropping nonfinite predictions.

    If any prediction entry is nonfinite on the evaluation set:
      relative_mse = NaN, MAE = NaN
    (optimistic bias from deleting failed voxels is forbidden).
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    if p.size != o.size:
        raise ValueError(f"pred/obs size mismatch: {p.size} vs {o.size}")
    n_nonfinite_pred = int(np.count_nonzero(~np.isfinite(p)))
    n_nonfinite_obs = int(np.count_nonzero(~np.isfinite(o)))
    out: dict[str, float] = {
        "n_values": int(p.size),
        "n_nonfinite_pred": float(n_nonfinite_pred),
        "n_nonfinite_obs": float(n_nonfinite_obs),
    }
    if n_nonfinite_pred > 0 or n_nonfinite_obs > 0:
        out["MAE"] = float("nan")
        out["relative_mse"] = float("nan")
        return out
    base = dwi_reconstruction_metrics(p, o, eps=eps)
    out["MAE"] = float(base["MAE"])
    out["relative_mse"] = float(base["relative_mse"])
    return out


@torch.no_grad()
def predict_dwi_wls(
    *,
    S0: np.ndarray,
    D: np.ndarray,
    flat_idx: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """WLS forward on selected flat voxels. Returns [V,N]."""
    idx = np.asarray(flat_idx, dtype=np.int64)
    S0_v = torch.from_numpy(np.ascontiguousarray(S0.reshape(-1)[idx], dtype=np.float32)).to(device)
    D_v = torch.from_numpy(np.ascontiguousarray(D.reshape(-1, 3, 3)[idx], dtype=np.float32)).to(device)
    bvals_t = torch.from_numpy(np.asarray(bvals, dtype=np.float32)).to(device)
    bvecs_t = torch.from_numpy(np.asarray(bvecs, dtype=np.float32)).to(device)
    pred = dti_forward_signal(S0_v, D_v, bvals_t, bvecs_t)
    return pred.detach().float().cpu().numpy()


@torch.no_grad()
def predict_dwi_inr(
    *,
    model: torch.nn.Module,
    shape_xyz: tuple[int, int, int],
    flat_idx: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    """INR forward on the same flat voxels. Returns [V,N]."""
    model.eval()
    idx = np.asarray(flat_idx, dtype=np.int64)
    coords = coords_from_flat_indices(shape_xyz, idx)
    bvals_t = torch.from_numpy(np.asarray(bvals, dtype=np.float32)).to(device)
    bvecs_t = torch.from_numpy(np.asarray(bvecs, dtype=np.float32)).to(device)
    preds: list[np.ndarray] = []
    for i in range(0, idx.size, int(chunk)):
        sl = slice(i, min(i + int(chunk), idx.size))
        xyz = torch.from_numpy(coords[sl]).to(device)
        S0, D = model(xyz)
        pred = dti_forward_signal(S0, D, bvals_t, bvecs_t)
        preds.append(pred.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, int(np.asarray(bvals).size)), dtype=np.float32)


def count_nonfinite_entries(arr: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isfinite(np.asarray(arr))))


def evaluate_wls_inr_dwi_common(
    *,
    observed_dwi: np.ndarray,
    brain_mask: np.ndarray,
    wls_valid_mask: np.ndarray,
    wls_S0: np.ndarray,
    wls_D: np.ndarray,
    model: torch.nn.Module,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    device: torch.device,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    seed: int = DEFAULT_EVAL_SEED,
) -> dict[str, Any]:
    """
    Single shared voxel index set → WLS and INR DWI RelMSE on common mask.

    Pred nonfinite does not shrink the mask; RelMSE becomes NaN instead.
    """
    dwi = np.ascontiguousarray(observed_dwi, dtype=np.float32)
    common = build_common_dwi_eval_mask(brain_mask, wls_valid_mask, dwi)
    n_common = int(np.count_nonzero(common))
    flat_idx, coords, n_eval_full = sample_eval_voxel_indices(
        common, max_voxels=max_voxels, seed=seed
    )
    n_sampled = int(flat_idx.size)
    n_vol = int(dwi.shape[-1])
    shape_xyz = (int(dwi.shape[0]), int(dwi.shape[1]), int(dwi.shape[2]))

    if n_sampled == 0:
        return {
            "n_common_dwi_voxels": n_common,
            "n_eval_dwi_voxels": n_eval_full,
            "n_sampled_voxels": 0,
            "n_common_dwi_values": 0,
            "WLS_DWI_RelMSE": float("nan"),
            "INR_DWI_RelMSE": float("nan"),
            "Delta": float("nan"),
            "Ratio": float("nan"),
            "WLS_DWI_MAE": float("nan"),
            "INR_DWI_MAE": float("nan"),
            "n_wls_nonfinite": 0,
            "n_inr_nonfinite": 0,
            "eval_seed": int(seed),
            "max_voxels": int(max_voxels),
            "common_mask_definition": COMMON_MASK_DEF,
        }

    obs = dwi.reshape(-1, n_vol)[flat_idx]
    # Shared indices: WLS and INR evaluated on identical voxels.
    pred_wls = predict_dwi_wls(
        S0=wls_S0, D=wls_D, flat_idx=flat_idx, bvals=bvals, bvecs=bvecs, device=device
    )
    # Use precomputed coords (same order as flat_idx) for INR
    model.eval()
    bvals_t = torch.from_numpy(np.asarray(bvals, dtype=np.float32)).to(device)
    bvecs_t = torch.from_numpy(np.asarray(bvecs, dtype=np.float32)).to(device)
    preds: list[np.ndarray] = []
    chunk = 65536
    with torch.no_grad():
        for i in range(0, n_sampled, chunk):
            sl = slice(i, min(i + chunk, n_sampled))
            xyz = torch.from_numpy(coords[sl]).to(device)
            S0, D = model(xyz)
            preds.append(dti_forward_signal(S0, D, bvals_t, bvecs_t).detach().float().cpu().numpy())
    pred_inr = np.concatenate(preds, axis=0)

    n_wls_nf = count_nonfinite_entries(pred_wls)
    n_inr_nf = count_nonfinite_entries(pred_inr)
    m_wls = dwi_metrics_no_silent_drop(pred_wls, obs)
    m_inr = dwi_metrics_no_silent_drop(pred_inr, obs)
    wls_r = float(m_wls["relative_mse"])
    inr_r = float(m_inr["relative_mse"])
    delta = inr_r - wls_r if np.isfinite(inr_r) and np.isfinite(wls_r) else float("nan")
    ratio = (inr_r / wls_r) if np.isfinite(inr_r) and np.isfinite(wls_r) and wls_r > 0 else float("nan")

    return {
        "n_common_dwi_voxels": n_common,
        "n_eval_dwi_voxels": n_eval_full,
        "n_sampled_voxels": n_sampled,
        "n_common_dwi_values": int(n_sampled * n_vol),
        "WLS_DWI_RelMSE": wls_r,
        "INR_DWI_RelMSE": inr_r,
        "Delta": float(delta),
        "Ratio": float(ratio),
        "WLS_DWI_MAE": float(m_wls["MAE"]),
        "INR_DWI_MAE": float(m_inr["MAE"]),
        "n_wls_nonfinite": n_wls_nf,
        "n_inr_nonfinite": n_inr_nf,
        "eval_seed": int(seed),
        "max_voxels": int(max_voxels),
        "common_mask_definition": COMMON_MASK_DEF,
        "flat_idx": flat_idx,
        "shape_xyz": shape_xyz,
    }
