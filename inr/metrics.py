"""Legacy helpers — prefer metrics_schema for new code."""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics_schema import mae, parameter_agreement_vs_wls, pearson_r, rmse


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    y = np.asarray(b, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float("nan") if x.size == 0 else float(np.mean((x - y) ** 2))


def compare_dti_maps(
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Backward-compatible flat dict; new code should use parameter_agreement_vs_wls. """
    nested = parameter_agreement_vs_wls(pred, ref, mask)
    out: dict[str, Any] = {"n_voxels": nested["n_voxels"]}
    for key in ("FA", "MD", "AD", "RD"):
        out[f"{key}_mae"] = nested[key]["MAE"]
        out[f"{key}_mse"] = float(nested[key]["RMSE"] ** 2) if nested[key]["RMSE"] == nested[key]["RMSE"] else float("nan")
        out[f"{key}_rmse"] = nested[key]["RMSE"]
        out[f"{key}_pearson"] = nested[key]["Pearson"]
    return out


__all__ = ["mae", "mse", "rmse", "pearson_r", "compare_dti_maps", "parameter_agreement_vs_wls"]
