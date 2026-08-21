"""Model-independent common-mask DWI evaluation (Independent INR baseline).

Primary common mask (fixed before any INR/WLS prediction):
  brain_mask AND WLS_valid_mask AND finite(observed_DWI over volumes)

Prediction finiteness is NOT part of the common mask (avoids optimistic bias).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .coords import voxel_coords_normalized
from .metrics_schema import dwi_reconstruction_metrics
from .physics import dti_forward_signal

DEFAULT_MAX_VOXELS = 131072
DEFAULT_EVAL_SEED = 42


def build_common_dwi_eval_mask(
    brain_mask: np.ndarray,
    wls_valid_mask: np.ndarray,
    observed_dwi: np.ndarray,
) -> np.ndarray:
    """
    Model-independent common mask.

    observed_dwi: [X,Y,Z,N] — a voxel is finite-obs iff ALL volumes are finite.
    """
    brain = np.asarray(brain_mask, dtype=bool)
    valid = np.asarray(wls_valid_mask, dtype=bool)
    obs = np.asarray(observed_dwi)
    if obs.ndim != 4:
        raise ValueError(f"observed_dwi must be 4D [X,Y,Z,N], got {obs.shape}")
    if brain.shape != obs.shape[:3] or valid.shape != obs.shape[:3]:
        raise ValueError(
            f"mask/data spatial mismatch: brain={brain.shape} valid={valid.shape} dwi={obs.shape[:3]}"
        )
    finite_obs = np.all(np.isfinite(obs), axis=-1)
    return brain & valid & finite_obs


def sample_eval_voxel_indices(
    common_mask: np.ndarray,
    *,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    seed: int = DEFAULT_EVAL_SEED,
) -> np.ndarray:
    """Deterministic flat indices into X*Y*Z. Same policy for WLS and INR."""
    flat = np.flatnonzero(np.asarray(common_mask, dtype=bool).reshape(-1))
    if flat.size == 0:
        return flat.astype(np.int64)
    max_voxels = int(max_voxels)
    if flat.size <= max_voxels:
        return flat.astype(np.int64)
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(flat.size, size=max_voxels, replace=False)
    return np.sort(flat[sel].astype(np.int64))


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
    flat_idx = sample_eval_voxel_indices(common, max_voxels=max_voxels, seed=seed)
    n_eval = int(flat_idx.size)
    n_vol = int(dwi.shape[-1])
    shape_xyz = (int(dwi.shape[0]), int(dwi.shape[1]), int(dwi.shape[2]))

    if n_eval == 0:
        return {
            "n_common_dwi_voxels": 0,
            "n_eval_dwi_voxels": 0,
            "n_common_dwi_values": 0,
            "WLS_DWI_RelMSE_common": float("nan"),
            "WLS_DWI_MAE_common": float("nan"),
            "INR_DWI_RelMSE_common": float("nan"),
            "INR_DWI_MAE_common": float("nan"),
            "n_wls_nonfinite": 0,
            "n_inr_nonfinite": 0,
            "eval_seed": int(seed),
            "max_voxels": int(max_voxels),
            "common_mask_definition": "brain AND WLS_valid AND finite(observed_DWI)",
        }

    obs = dwi.reshape(-1, n_vol)[flat_idx]
    pred_wls = predict_dwi_wls(
        S0=wls_S0, D=wls_D, flat_idx=flat_idx, bvals=bvals, bvecs=bvecs, device=device
    )
    pred_inr = predict_dwi_inr(
        model=model,
        shape_xyz=shape_xyz,
        flat_idx=flat_idx,
        bvals=bvals,
        bvecs=bvecs,
        device=device,
    )

    n_wls_nf = count_nonfinite_entries(pred_wls)
    n_inr_nf = count_nonfinite_entries(pred_inr)
    m_wls = dwi_metrics_no_silent_drop(pred_wls, obs)
    m_inr = dwi_metrics_no_silent_drop(pred_inr, obs)

    return {
        "n_common_dwi_voxels": n_common,
        "n_eval_dwi_voxels": n_eval,
        "n_common_dwi_values": int(n_eval * n_vol),
        "WLS_DWI_RelMSE_common": float(m_wls["relative_mse"]),
        "WLS_DWI_MAE_common": float(m_wls["MAE"]),
        "INR_DWI_RelMSE_common": float(m_inr["relative_mse"]),
        "INR_DWI_MAE_common": float(m_inr["MAE"]),
        "n_wls_nonfinite": n_wls_nf,
        "n_inr_nonfinite": n_inr_nf,
        "eval_seed": int(seed),
        "max_voxels": int(max_voxels),
        "common_mask_definition": "brain AND WLS_valid AND finite(observed_DWI)",
        "flat_idx": flat_idx,
    }
