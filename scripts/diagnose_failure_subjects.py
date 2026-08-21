#!/usr/bin/env python
"""29-subject Failure Diagnosis: Raw DWI → WLS → INR attribution.

Does NOT modify INR architecture, training, physics, or WLS fitting.

Protocol (must match official Independent INR eval / G2):
  common_mask = brain & WLS_valid
  shared sample: seed=42, max_voxels=131072
  RelMSE = ||pred-obs||^2 / (||obs||^2 + eps)
  no ad-hoc S0 threshold (no 1 < S0 < 15000)

Usage:
  python scripts/diagnose_failure_subjects.py
  python scripts/diagnose_failure_subjects.py --recompute-dwi

Outputs (does not touch eval_common_mask/summary.csv):
  outputs/v1_schema_train/independent_inr/failure_diagnosis/
    summary.csv
    aggregate.md
    wls_vs_inr_dwi_relmse.png
    wls_vs_inr_parameter_agreement.png
    112920.md / 124422.md / 130720.md  (focus subjects; also any auto non-normal)
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.common_dwi_eval import (  # noqa: E402
    COMMON_MASK_DEF,
    DEFAULT_EVAL_SEED,
    DEFAULT_MAX_VOXELS,
    build_common_dwi_eval_mask,
    evaluate_wls_inr_dwi_common,
)
from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402
from inr.metrics_schema import PARAM_KEYS, parameter_agreement_vs_wls  # noqa: E402
from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import resolve_device  # noqa: E402

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit(f"nibabel required: {e}") from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(f"matplotlib required: {e}") from e

FOCUS_SUBJECTS = ("112920", "124422", "130720")

SUMMARY_FIELDS = [
    "subject_id",
    "status",
    "outlier_candidate",
    "status_reason",
    "n_brain_voxels",
    "n_wls_valid_voxels",
    "n_common_voxels",
    "n_sampled_voxels",
    "WLS_DWI_RelMSE",
    "INR_DWI_RelMSE",
    "DWI_RelMSE_Delta",
    "DWI_RelMSE_Ratio",
    "WLS_FA_mean",
    "WLS_FA_std",
    "INR_FA_mean",
    "INR_FA_std",
    "WLS_MD_mean",
    "WLS_MD_std",
    "INR_MD_mean",
    "INR_MD_std",
    "WLS_AD_mean",
    "WLS_AD_std",
    "INR_AD_mean",
    "INR_AD_std",
    "WLS_RD_mean",
    "WLS_RD_std",
    "INR_RD_mean",
    "INR_RD_std",
    "INR_FA_MAE",
    "INR_FA_RMSE",
    "INR_FA_Pearson",
    "INR_MD_MAE",
    "INR_MD_RMSE",
    "INR_MD_Pearson",
    "INR_AD_MAE",
    "INR_AD_RMSE",
    "INR_AD_Pearson",
    "INR_RD_MAE",
    "INR_RD_RMSE",
    "INR_RD_Pearson",
    "raw_dwi_quality",
    "n_total_volumes",
    "n_b0_volumes",
    "n_b1000_volumes",
    "raw_finite_ratio",
    "raw_signal_min",
    "raw_signal_max",
    "raw_signal_mean",
    "raw_signal_std",
    "flag_wls_dwi_high",
    "flag_inr_dwi_high",
    "flag_fa_mae_high",
    "flag_wls_fa_abnormal",
    "robust_z_WLS_DWI",
    "robust_z_INR_DWI",
    "robust_z_FA_MAE",
    "common_mask_definition",
    "eval_seed",
    "max_voxels",
    "dwi_source",
]


def _nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _wls_D(trad: Path) -> np.ndarray:
    D = _nii(trad / "D.nii.gz").astype(np.float32)
    if D.ndim == 5 and D.shape[-2:] == (3, 3):
        return D
    if D.ndim == 4 and D.shape[-1] == 9:
        return D.reshape(*D.shape[:3], 3, 3)
    raise ValueError(f"unexpected D shape {D.shape}")


def _load_inr(ckpt: Path, device) -> SpatialDTIINR:
    import torch

    obj = torch.load(ckpt, map_location=device, weights_only=False)
    c = obj.get("config", {})
    model = SpatialDTIINR(
        hidden=int(c.get("hidden", 128)),
        layers=int(c.get("layers", 4)),
        pe_freqs=int(c.get("pe_freqs", 8)),
    ).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std())


def _agg(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("n", "mean", "median", "std", "q1", "q3", "iqr")}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": float(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _mad(a: np.ndarray) -> float:
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _robust_z(x: float, cohort: np.ndarray) -> float:
    a = np.asarray(cohort, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3 or not np.isfinite(x):
        return float("nan")
    med = float(np.median(a))
    mad = _mad(a)
    if mad <= 0:
        return 0.0 if abs(x - med) < 1e-12 else float("inf")
    return float((x - med) / (1.4826 * mad))


def _iqr_high(x: float, cohort: np.ndarray, k: float = 1.5) -> bool:
    """True if x > Q3 + k*IQR (upper outlier fence)."""
    a = np.asarray(cohort, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 5 or not np.isfinite(x):
        return False
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return bool(x > q3)
    return bool(x > q3 + k * iqr)


def _iqr_outside(x: float, cohort: np.ndarray, k: float = 1.5) -> bool:
    a = np.asarray(cohort, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 5 or not np.isfinite(x):
        return False
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return bool(x != q1)
    return bool(x < q1 - k * iqr or x > q3 + k * iqr)


def _raw_dwi_quality(dwi: np.ndarray, bvals: np.ndarray, b0_threshold: float, shell_tol: float) -> dict[str, Any]:
    b = np.asarray(bvals, dtype=np.float64).ravel()
    x = np.asarray(dwi, dtype=np.float64)
    finite = np.isfinite(x)
    n_fin = int(finite.sum())
    n_tot = int(x.size)
    vals = x[finite]
    return {
        "raw_dwi_quality": "available",
        "n_total_volumes": int(b.size),
        "n_b0_volumes": int(np.count_nonzero(b < b0_threshold)),
        "n_b1000_volumes": int(np.count_nonzero((b >= b0_threshold) & (np.abs(b - 1000.0) <= shell_tol))),
        "raw_finite_ratio": float(n_fin / max(n_tot, 1)),
        "raw_signal_min": float(vals.min()) if vals.size else float("nan"),
        "raw_signal_max": float(vals.max()) if vals.size else float("nan"),
        "raw_signal_mean": float(vals.mean()) if vals.size else float("nan"),
        "raw_signal_std": float(vals.std()) if vals.size else float("nan"),
    }


def _load_g2(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r.get("Subject") or r.get("subject_id")
            if not sid:
                continue
            out[sid] = {
                "WLS_DWI_RelMSE": float(r["WLS_DWI_RelMSE"]),
                "INR_DWI_RelMSE": float(r["INR_DWI_RelMSE"]),
                "DWI_RelMSE_Delta": float(r.get("Delta", float(r["INR_DWI_RelMSE"]) - float(r["WLS_DWI_RelMSE"]))),
                "DWI_RelMSE_Ratio": float(r.get("Ratio", float("nan"))),
                "n_sampled_voxels": float(r.get("n_sampled_voxels", DEFAULT_MAX_VOXELS)),
                "n_common_from_g2": float(r.get("n_eval_voxels", float("nan"))),
            }
    return out


def diagnose_subject(
    sid: str,
    *,
    cfg: dict,
    exp: Path,
    trad_root: Path,
    device,
    g2: dict[str, dict[str, float]] | None,
    recompute_dwi: bool,
) -> dict[str, Any]:
    import torch

    trad = trad_root / sid
    ckpt = exp / sid / "best.pt"
    maps_path = exp / sid / "maps.npz"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not maps_path.is_file():
        raise FileNotFoundError(maps_path)

    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
    bvals_full = bundle["bvals"]
    bvecs_full = normalize_bvecs(bvals_full, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
    vol_m = shell_volume_mask(
        bvals_full,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    dwi = np.ascontiguousarray(bundle["data"][..., vol_m], dtype=np.float32)
    brain = np.asarray(bundle["brain_mask"], dtype=bool)
    valid = np.asarray(_nii(trad / "valid_mask.nii.gz") > 0.5)
    common = build_common_dwi_eval_mask(brain, valid, dwi)

    n_brain = int(brain.sum())
    n_valid = int(valid.sum())
    n_common = int(common.sum())

    raw_q = _raw_dwi_quality(
        dwi,
        bvals_full[vol_m],
        float(cfg["b0_threshold"]),
        float(cfg["shell_tol"]),
    )

    S0 = np.ascontiguousarray(_nii(trad / "S0.nii.gz"), dtype=np.float32)
    D = np.ascontiguousarray(_wls_D(trad))
    FA_w = _nii(trad / "FA.nii.gz").astype(np.float32)
    MD_w = _nii(trad / "MD.nii.gz").astype(np.float32)
    AD_w = _nii(trad / "AD.nii.gz").astype(np.float32)
    RD_w = _nii(trad / "RD.nii.gz").astype(np.float32)

    npz = np.load(maps_path)
    FA_i, MD_i, AD_i, RD_i = npz["FA"], npz["MD"], npz["AD"], npz["RD"]
    pred = {"FA": FA_i, "MD": MD_i, "AD": AD_i, "RD": RD_i}
    ref = {"FA": FA_w, "MD": MD_w, "AD": AD_w, "RD": RD_w}
    agree = parameter_agreement_vs_wls(pred, ref, common)

    row: dict[str, Any] = {
        "subject_id": sid,
        "n_brain_voxels": n_brain,
        "n_wls_valid_voxels": n_valid,
        "n_common_voxels": n_common,
        "common_mask_definition": COMMON_MASK_DEF,
        "eval_seed": DEFAULT_EVAL_SEED,
        "max_voxels": DEFAULT_MAX_VOXELS,
        **raw_q,
    }
    for key in PARAM_KEYS:
        wm, ws = _mean_std(ref[key][common])
        im, iss = _mean_std(pred[key][common])
        row[f"WLS_{key}_mean"] = wm
        row[f"WLS_{key}_std"] = ws
        row[f"INR_{key}_mean"] = im
        row[f"INR_{key}_std"] = iss
        row[f"INR_{key}_MAE"] = float(agree[key]["MAE"])
        row[f"INR_{key}_RMSE"] = float(agree[key]["RMSE"])
        row[f"INR_{key}_Pearson"] = float(agree[key]["Pearson"])

    use_g2 = (not recompute_dwi) and g2 is not None and sid in g2
    if use_g2:
        g = g2[sid]
        row["WLS_DWI_RelMSE"] = g["WLS_DWI_RelMSE"]
        row["INR_DWI_RelMSE"] = g["INR_DWI_RelMSE"]
        row["DWI_RelMSE_Delta"] = g["DWI_RelMSE_Delta"]
        row["DWI_RelMSE_Ratio"] = g["DWI_RelMSE_Ratio"]
        row["n_sampled_voxels"] = int(g["n_sampled_voxels"])
        row["dwi_source"] = "g2_shared_indices_reuse"
    else:
        model = _load_inr(ckpt, device)
        ev = evaluate_wls_inr_dwi_common(
            observed_dwi=dwi,
            brain_mask=brain,
            wls_valid_mask=valid,
            wls_S0=S0,
            wls_D=D,
            model=model,
            bvals=bvals_full[vol_m],
            bvecs=bvecs_full[vol_m],
            device=device,
            max_voxels=DEFAULT_MAX_VOXELS,
            seed=DEFAULT_EVAL_SEED,
        )
        row["WLS_DWI_RelMSE"] = float(ev["WLS_DWI_RelMSE"])
        row["INR_DWI_RelMSE"] = float(ev["INR_DWI_RelMSE"])
        row["DWI_RelMSE_Delta"] = float(ev["Delta"])
        row["DWI_RelMSE_Ratio"] = float(ev["Ratio"])
        row["n_sampled_voxels"] = int(ev["n_sampled_voxels"])
        row["dwi_source"] = "recompute_evaluate_wls_inr_dwi_common"
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del bundle, dwi, npz
    gc.collect()
    return row


ROBUST_Z_ELEVATED = 2.5  # secondary elevation rule (distribution-based, not absolute RelMSE cut)


def classify_rows(rows: list[dict[str, Any]]) -> None:
    """Distribution-based flags + status. Mutates rows in place.

    Rules (documented in aggregate.md):
      high_*  := x > Q3 + 1.5*IQR  OR  robust_z >= 2.5  (cohort-relative)
      WLS FA abnormal := WLS_FA_mean outside [Q1-1.5IQR, Q3+1.5IQR]
      (WLS DWI RelMSE alone is NOT treated as sufficient for 'WLS failure'
       because rare S0_hat tails inflate global RelMSE under valid_mask.)

      status priority:
        1. data_or_wls_suspect: WLS DWI high AND WLS FA abnormal AND INR DWI high
        2. wls_difficult: WLS DWI high AND WLS FA abnormal AND INR DWI not high
        3. inr_specific: INR DWI high AND FA MAE high AND WLS FA NOT abnormal
        4. parameter_only: FA MAE high AND INR DWI not high
        5. normal: otherwise
    """
    wls_dwi = np.array([float(r["WLS_DWI_RelMSE"]) for r in rows], dtype=np.float64)
    inr_dwi = np.array([float(r["INR_DWI_RelMSE"]) for r in rows], dtype=np.float64)
    fa_mae = np.array([float(r["INR_FA_MAE"]) for r in rows], dtype=np.float64)
    wls_fa = np.array([float(r["WLS_FA_mean"]) for r in rows], dtype=np.float64)

    for r in rows:
        rz_w = _robust_z(float(r["WLS_DWI_RelMSE"]), wls_dwi)
        rz_i = _robust_z(float(r["INR_DWI_RelMSE"]), inr_dwi)
        rz_f = _robust_z(float(r["INR_FA_MAE"]), fa_mae)
        wh = _iqr_high(float(r["WLS_DWI_RelMSE"]), wls_dwi) or (
            np.isfinite(rz_w) and rz_w >= ROBUST_Z_ELEVATED
        )
        ih = _iqr_high(float(r["INR_DWI_RelMSE"]), inr_dwi) or (
            np.isfinite(rz_i) and rz_i >= ROBUST_Z_ELEVATED
        )
        fh = _iqr_high(float(r["INR_FA_MAE"]), fa_mae) or (
            np.isfinite(rz_f) and rz_f >= ROBUST_Z_ELEVATED
        )
        wa = _iqr_outside(float(r["WLS_FA_mean"]), wls_fa)

        r["flag_wls_dwi_high"] = int(wh)
        r["flag_inr_dwi_high"] = int(ih)
        r["flag_fa_mae_high"] = int(fh)
        r["flag_wls_fa_abnormal"] = int(wa)
        r["robust_z_WLS_DWI"] = rz_w
        r["robust_z_INR_DWI"] = rz_i
        r["robust_z_FA_MAE"] = rz_f

        if wh and wa and ih:
            status = "data_or_wls_suspect"
            reason = "WLS DWI + WLS FA mean + INR DWI all IQR-outlier; subject/data or WLS difficulty candidate"
        elif wh and wa and not ih:
            status = "wls_difficult"
            reason = "WLS DWI and WLS FA mean IQR-outlier while INR DWI not; do not attribute to INR alone"
        elif ih and fh and not wa:
            status = "inr_specific"
            reason = "INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure"
        elif fh and not ih:
            status = "parameter_only"
            reason = "FA MAE IQR-high but INR DWI not; reconstruction OK-ish, parameter agreement weak"
        elif ih and not fh:
            # elevated DWI without FA MAE fence — still INR-side signal
            status = "inr_specific"
            reason = "INR DWI IQR-high (FA MAE not fence-outlier); INR reconstruction candidate"
        elif fh and ih and wa:
            status = "data_or_wls_suspect"
            reason = "INR + FA MAE high together with abnormal WLS FA; mixed attribution"
        else:
            status = "normal"
            reason = "within cohort IQR fences on primary flags"

        r["status"] = status
        r["status_reason"] = reason
        r["outlier_candidate"] = int(status != "normal" or wh or ih or fh)


def plot_dwi_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [float(r["WLS_DWI_RelMSE"]) for r in rows]
    ys = [float(r["INR_DWI_RelMSE"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for r, x, y in zip(rows, xs, ys):
        sid = r["subject_id"]
        focus = sid in FOCUS_SUBJECTS
        ax.scatter(
            x,
            y,
            s=90 if focus else 45,
            c="#c0392b" if focus else "#2c3e50",
            zorder=3 if focus else 2,
            edgecolors="k" if focus else "none",
            linewidths=1.0 if focus else 0.0,
        )
        if focus or r["status"] != "normal":
            ax.annotate(sid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.05 * (hi - lo + 1e-9)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color="#7f8c8d", lw=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("WLS DWI RelMSE")
    ax.set_ylabel("INR DWI RelMSE")
    ax.set_title("WLS vs INR DWI RelMSE (shared common-mask voxels)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_param_agreement(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key in zip(axes.ravel(), PARAM_KEYS):
        for r in rows:
            wm = float(r[f"WLS_{key}_mean"])
            im = float(r[f"INR_{key}_mean"])
            sid = r["subject_id"]
            focus = sid in FOCUS_SUBJECTS
            ax.scatter(
                wm,
                im,
                s=70 if focus else 35,
                c="#c0392b" if focus else "#34495e",
                zorder=3 if focus else 2,
                edgecolors="k" if focus else "none",
            )
            if focus:
                ax.annotate(sid, (wm, im), textcoords="offset points", xytext=(4, 4), fontsize=7)
        vals_x = [float(r[f"WLS_{key}_mean"]) for r in rows]
        vals_y = [float(r[f"INR_{key}_mean"]) for r in rows]
        lo = min(min(vals_x), min(vals_y))
        hi = max(max(vals_x), max(vals_y))
        pad = 0.05 * (hi - lo + 1e-12)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, "--", color="#95a5a6", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"WLS {key} mean")
        ax.set_ylabel(f"INR {key} mean")
        ax.set_title(key)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Parameter map means on common_mask (WLS vs INR)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_subject_report(r: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Subject {r['subject_id']}",
        "",
        "## Raw / evaluation information",
        "",
        f"- brain voxels: {r['n_brain_voxels']}",
        f"- WLS valid voxels: {r['n_wls_valid_voxels']}",
        f"- common voxels (`{r['common_mask_definition']}`): {r['n_common_voxels']}",
        f"- sampled voxels (shared WLS/INR): {r['n_sampled_voxels']}",
        f"- eval seed / max_voxels: {r['eval_seed']} / {r['max_voxels']}",
        f"- DWI metric source: {r['dwi_source']}",
        "",
        "## Raw DWI quality (lightweight)",
        "",
        f"- raw_dwi_quality: {r['raw_dwi_quality']}",
        f"- n_total_volumes: {r['n_total_volumes']}",
        f"- n_b0_volumes: {r['n_b0_volumes']}",
        f"- n_b1000_volumes: {r['n_b1000_volumes']}",
        f"- finite_ratio: {float(r['raw_finite_ratio']):.6f}",
        f"- signal min/max/mean/std: {float(r['raw_signal_min']):.4g} / {float(r['raw_signal_max']):.4g} / "
        f"{float(r['raw_signal_mean']):.4g} / {float(r['raw_signal_std']):.4g}",
        "",
        "## DWI",
        "",
        f"- WLS DWI RelMSE: {float(r['WLS_DWI_RelMSE']):.6g}",
        f"- INR DWI RelMSE: {float(r['INR_DWI_RelMSE']):.6g}",
        f"- Delta (INR − WLS): {float(r['DWI_RelMSE_Delta']):.6g}",
        f"- Ratio (INR / WLS): {float(r['DWI_RelMSE_Ratio']):.6g}",
        f"- robust z (WLS / INR): {float(r['robust_z_WLS_DWI']):.3g} / {float(r['robust_z_INR_DWI']):.3g}",
        "",
        "## Parameters (common mask)",
        "",
        "| map | WLS mean±std | INR mean±std | MAE | RMSE | Pearson |",
        "|-----|-------------:|-------------:|----:|-----:|--------:|",
    ]
    for key in PARAM_KEYS:
        lines.append(
            f"| {key} | {float(r[f'WLS_{key}_mean']):.6g}±{float(r[f'WLS_{key}_std']):.6g} | "
            f"{float(r[f'INR_{key}_mean']):.6g}±{float(r[f'INR_{key}_std']):.6g} | "
            f"{float(r[f'INR_{key}_MAE']):.6g} | {float(r[f'INR_{key}_RMSE']):.6g} | "
            f"{float(r[f'INR_{key}_Pearson']):.4g} |"
        )
    lines += [
        "",
        "## Diagnosis",
        "",
        f"- automatic status: **{r['status']}**",
        f"- outlier_candidate: {bool(int(r['outlier_candidate']))}",
        f"- flags: wls_dwi_high={r['flag_wls_dwi_high']} inr_dwi_high={r['flag_inr_dwi_high']} "
        f"fa_mae_high={r['flag_fa_mae_high']} wls_fa_abnormal={r['flag_wls_fa_abnormal']}",
        f"- reason: {r['status_reason']}",
        "",
        "This is an automated *candidate* attribution, not a final scientific conclusion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_aggregate(rows: list[dict[str, Any]], path: Path, *, iqr_k: float = 1.5) -> None:
    wls_a = _agg([float(r["WLS_DWI_RelMSE"]) for r in rows])
    inr_a = _agg([float(r["INR_DWI_RelMSE"]) for r in rows])
    ratio_a = _agg([float(r["DWI_RelMSE_Ratio"]) for r in rows])
    fa_a = _agg([float(r["INR_FA_MAE"]) for r in rows])

    non_normal = [r for r in rows if r["status"] != "normal"]
    focus_rows = [r for r in rows if r["subject_id"] in FOCUS_SUBJECTS]

    lines = [
        "# Failure Diagnosis Aggregate (Independent INR, 29 subjects)",
        "",
        "## Protocol",
        "",
        f"- common_mask = `{COMMON_MASK_DEF}` (reuse `valid_mask` / `s0_ok` from existing WLS fit; no new S0 threshold)",
        f"- shared sampled voxels: seed={DEFAULT_EVAL_SEED}, max_voxels={DEFAULT_MAX_VOXELS}",
        "- WLS and INR DWI RelMSE use identical indices via `evaluate_wls_inr_dwi_common` / G2 reuse",
        "- RelMSE = Σ(pred−obs)² / Σ(obs)²",
        "- **No model / training / dti_fit changes** — diagnosis only",
        "",
        "## Outlier rule (not a fixed absolute threshold)",
        "",
        f"- Upper fence: `x > Q3 + {iqr_k}·IQR` on the 29-subject cohort",
        f"- OR elevated if robust z ≥ {ROBUST_Z_ELEVATED} "
        f"(robust z = `(x − median) / (1.4826 · MAD)`)",
        "- `WLS_FA_mean` abnormal: outside `[Q1−k·IQR, Q3+k·IQR]`",
        "- Note: high `WLS_DWI_RelMSE` alone can reflect rare exploding `S0_hat` under `S0 < 1e6`; "
        "classification therefore also checks WLS FA mean fences and INR FA MAE.",
        "",
        "## Overall",
        "",
        f"- N subjects = {len(rows)}",
        "",
        "### WLS DWI RelMSE",
        f"- mean={wls_a['mean']:.6g}  median={wls_a['median']:.6g}  std={wls_a['std']:.6g}",
        f"- Q1={wls_a['q1']:.6g}  Q3={wls_a['q3']:.6g}  IQR={wls_a['iqr']:.6g}",
        "",
        "### INR DWI RelMSE",
        f"- mean={inr_a['mean']:.6g}  median={inr_a['median']:.6g}  std={inr_a['std']:.6g}",
        f"- Q1={inr_a['q1']:.6g}  Q3={inr_a['q3']:.6g}  IQR={inr_a['iqr']:.6g}",
        "",
        "### INR/WLS ratio",
        f"- mean={ratio_a['mean']:.6g}  median={ratio_a['median']:.6g}",
        "",
        "### INR FA MAE (vs WLS)",
        f"- mean={fa_a['mean']:.6g}  median={fa_a['median']:.6g}  std={fa_a['std']:.6g}",
        "",
        "## Status counts",
        "",
    ]
    from collections import Counter

    cnt = Counter(r["status"] for r in rows)
    for k in ("normal", "inr_specific", "parameter_only", "wls_difficult", "data_or_wls_suspect"):
        lines.append(f"- {k}: {cnt.get(k, 0)}")

    lines += ["", "## Failure candidates (auto + focus)", ""]
    lines.append("| subject | status | WLS RelMSE | INR RelMSE | FA MAE | WLS FA mean | INR FA mean | reason |")
    lines.append("|---------|--------|-----------:|-----------:|-------:|------------:|------------:|--------|")
    shown = {r["subject_id"] for r in non_normal} | set(FOCUS_SUBJECTS)
    for r in rows:
        if r["subject_id"] not in shown:
            continue
        lines.append(
            f"| {r['subject_id']} | {r['status']} | {float(r['WLS_DWI_RelMSE']):.4f} | "
            f"{float(r['INR_DWI_RelMSE']):.4f} | {float(r['INR_FA_MAE']):.4f} | "
            f"{float(r['WLS_FA_mean']):.4f} | {float(r['INR_FA_mean']):.4f} | {r['status_reason']} |"
        )

    lines += [
        "",
        "## Focus subjects (112920 / 124422 / 130720)",
        "",
    ]
    for r in focus_rows:
        lines.append(
            f"- **{r['subject_id']}**: status=`{r['status']}` — {r['status_reason']} "
            f"(INR RelMSE={float(r['INR_DWI_RelMSE']):.4f}, FA MAE={float(r['INR_FA_MAE']):.4f}, "
            f"robust_z_INR_DWI={float(r['robust_z_INR_DWI']):.2f}, robust_z_FA_MAE={float(r['robust_z_FA_MAE']):.2f})"
        )

    lines += [
        "",
        "## Interpretation (candidates only)",
        "",
        "- **subject/data difficulty**: raw intensity / volume counts far from cohort, or mixed WLS+INR fences.",
        "- **WLS difficulty**: WLS DWI + WLS FA mean fences without INR DWI fence.",
        "- **INR-specific failure**: INR DWI / FA MAE fences while WLS FA mean stays in cohort.",
        "- **parameter-only failure**: FA MAE fence without INR DWI fence.",
        "",
        "Do not treat this table as a final paper claim — use it to decide whether Independent INR can freeze.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="29-subject failure diagnosis (no training changes)")
    ap.add_argument("--recompute-dwi", action="store_true", help="Recompute WLS/INR DWI instead of reusing G2 CSV")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--subject", default="")
    args = ap.parse_args()

    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    out = exp / "failure_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    trad_root = project_root() / "outputs" / "step1_traditional_dti"

    g2_path = exp / "eval_common_mask" / "g2_wls_vs_inr_dwi.csv"
    g2 = _load_g2(g2_path) if g2_path.is_file() and not args.recompute_dwi else None
    if g2 is None and not args.recompute_dwi and not g2_path.is_file():
        print("[diag] G2 CSV missing → will recompute DWI RelMSE")
    elif g2 is not None:
        print(f"[diag] reusing G2 DWI RelMSE from {g2_path}")
    else:
        print("[diag] --recompute-dwi: evaluating shared-voxel WLS/INR RelMSE")

    summary = exp / "eval_common_mask" / "summary.csv"
    if not summary.is_file():
        summary = exp / "summary.csv"
    if args.subject.strip():
        sids = [args.subject.strip()]
    elif summary.is_file():
        with open(summary, newline="", encoding="utf-8") as f:
            sids = [r["subject_id"] for r in csv.DictReader(f)]
    else:
        sids = sorted(p.parent.name for p in exp.glob("*/best.pt"))
    if int(args.max_subjects) > 0:
        sids = sids[: int(args.max_subjects)]

    device = resolve_device("auto")
    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(sids, 1):
        print(f"[{i}/{len(sids)}] diagnose {sid}")
        row = diagnose_subject(
            sid,
            cfg=cfg,
            exp=exp,
            trad_root=trad_root,
            device=device,
            g2=g2,
            recompute_dwi=bool(args.recompute_dwi) or g2 is None,
        )
        rows.append(row)
        print(
            f"    WLS={row['WLS_DWI_RelMSE']:.4f} INR={row['INR_DWI_RelMSE']:.4f} "
            f"FA_MAE={row['INR_FA_MAE']:.4f} FA_mean WLS/INR="
            f"{row['WLS_FA_mean']:.3f}/{row['INR_FA_mean']:.3f}"
        )

    classify_rows(rows)

    csv_path = out / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})

    save_json(out / "summary.json", {"n_subjects": len(rows), "rows": rows})
    write_aggregate(rows, out / "aggregate.md")
    plot_dwi_scatter(rows, out / "wls_vs_inr_dwi_relmse.png")
    plot_param_agreement(rows, out / "wls_vs_inr_parameter_agreement.png")

    report_ids = set(FOCUS_SUBJECTS) | {r["subject_id"] for r in rows if r["status"] != "normal"}
    for r in rows:
        if r["subject_id"] in report_ids:
            write_subject_report(r, out / f"{r['subject_id']}.md")

    print(f"\n[diag] wrote {csv_path}")
    print(f"[diag] wrote {out / 'aggregate.md'}")
    print(f"[diag] plots + per-subject reports → {out}")
    for r in rows:
        if r["subject_id"] in FOCUS_SUBJECTS or r["status"] != "normal":
            print(f"  {r['subject_id']}: {r['status']}  (outlier={r['outlier_candidate']})")


if __name__ == "__main__":
    main()
