#!/usr/bin/env python
"""3-subject failure diagnosis: data quality + WLS vs INR.

Subjects: 112920, 124422, 130720 (+ control 103515 for reference).

Uses common_mask = brain & WLS_valid and G2 shared-voxel DWI RelMSE.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.hcp_io import load_hcp_subject, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit(f"nibabel required: {e}") from e

FAILS = ("112920", "124422", "130720")
CONTROL = "103515"
ALL = (*FAILS, CONTROL)


def _nii(p: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(p)).dataobj)


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p50", "p99", "max")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def _load_g2(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["Subject"]: r for r in csv.DictReader(f)}


def _load_summary(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["subject_id"]: r for r in csv.DictReader(f)}


def diagnose_one(sid: str, cfg: dict, g2: dict, sm: dict) -> dict:
    trad = project_root() / "outputs" / "step1_traditional_dti" / sid
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    maps_path = exp / sid / "maps.npz"
    if not maps_path.is_file():
        maps_path = exp / "eval_common_mask" / sid / "maps.npz"  # may not exist
    # prefer training maps.npz
    maps_path = exp / sid / "maps.npz"

    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
    bvals = bundle["bvals"]
    vol_m = shell_volume_mask(
        bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    dwi = np.ascontiguousarray(bundle["data"][..., vol_m], dtype=np.float32)
    brain = np.asarray(bundle["brain_mask"], dtype=bool)
    valid = np.asarray(_nii(trad / "valid_mask.nii.gz") > 0.5)
    common = brain & valid
    n_brain = int(brain.sum())
    n_valid = int(valid.sum())
    n_common = int(common.sum())

    b0_m = bvals[vol_m] < float(cfg["b0_threshold"])
    dw_m = ~b0_m
    obs_b0 = dwi[..., b0_m].mean(axis=-1)
    obs_dw = dwi[..., dw_m]  # all b~1000

    S0 = _nii(trad / "S0.nii.gz").astype(np.float32)
    FA_w = _nii(trad / "FA.nii.gz").astype(np.float32)
    MD_w = _nii(trad / "MD.nii.gz").astype(np.float32)
    AD_w = _nii(trad / "AD.nii.gz").astype(np.float32)
    RD_w = _nii(trad / "RD.nii.gz").astype(np.float32)

    npz = np.load(maps_path)
    FA_i, MD_i, AD_i, RD_i = npz["FA"], npz["MD"], npz["AD"], npz["RD"]

    c = common
    s0_c = S0[c]
    st_s0 = _stats(s0_c)
    st_b0 = _stats(obs_b0[c])
    st_dw = _stats(obs_dw[c])
    # fraction extreme S0_hat (diagnostic only; not a new eval threshold)
    frac_s0_gt_1e4 = float(np.mean(s0_c > 1e4))
    frac_s0_gt_1e5 = float(np.mean(s0_c > 1e5))

    g = g2[sid]
    s = sm[sid]

    # WLS map means on common
    wls_fa_mean = float(FA_w[c].mean())
    wls_md_mean = float(MD_w[c].mean())
    wls_ad_mean = float(AD_w[c].mean())
    wls_rd_mean = float(RD_w[c].mean())
    inr_fa_mean = float(FA_i[c].mean())
    inr_md_mean = float(MD_i[c].mean())
    inr_ad_mean = float(AD_i[c].mean())
    inr_rd_mean = float(RD_i[c].mean())

    wls_rel = float(g["WLS_DWI_RelMSE"])
    inr_rel = float(g["INR_DWI_RelMSE"])
    fa_mae = float(s["FA_MAE"])
    md_mae = float(s["MD_MAE"])
    ad_mae = float(s["AD_MAE"])
    rd_mae = float(s["RD_MAE"])

    # Heuristic case label (experimental, for triage only)
    # Compare to control-like WLS RelMSE is hard due to outliers; use FA_MAE and INR RelMSE.
    cohort_inr_med = 0.07  # approx from G2
    if fa_mae >= 0.30 and inr_rel >= 0.12:
        if wls_rel >= 0.20 and frac_s0_gt_1e5 > 1e-5:
            case = "mixed_WLS_outlier_AND_INR_param_fail"
        elif inr_rel > wls_rel * 0.8 and fa_mae >= 0.30:
            # INR DWI not much worse than inflated WLS, but FA collapsed
            case = "INR_parameter_failure_signal_ambiguous"
        else:
            case = "INR_failure"
    else:
        case = "control_or_stable"

    # Refine with clearer rules for the three known fails:
    # - INR FA MAE catastrophic (~0.46)
    # - INR DWI RelMSE elevated vs typical ~0.05-0.08
    # - WLS RelMSE also high due to S0_hat tails (protocol artifact), not necessarily "WLS can't fit DTI"
    if sid in FAILS:
        if sid == "130720" and md_mae > 0.001:
            case = "INR_failure_eigenvalue_blowup"
        elif float(s.get("best_epoch", 200) or 200) < 100 and float(s.get("best_loss", 0)) > 0.3:
            case = "INR_optimization_collapse"
        else:
            case = "INR_reconstruction_and_FA_failure"
        # note WLS global RelMSE is not a clean data-quality score under valid_mask S0<1e6

    return {
        "subject": sid,
        "role": "failure" if sid in FAILS else "control",
        "N_brain": n_brain,
        "N_WLS_valid": n_valid,
        "N_common": n_common,
        "valid_over_brain": n_valid / max(n_brain, 1),
        # Raw DWI (common mask)
        "raw_b0_mean": st_b0["mean"],
        "raw_b0_std": st_b0["std"],
        "raw_b1000_mean": st_dw["mean"],
        "raw_b1000_std": st_dw["std"],
        "raw_b1000_p99": st_dw["p99"],
        # S0
        "S0_hat_mean": st_s0["mean"],
        "S0_hat_p50": st_s0["p50"],
        "S0_hat_p99": st_s0["p99"],
        "S0_hat_max": st_s0["max"],
        "frac_S0_gt_1e4": frac_s0_gt_1e4,
        "frac_S0_gt_1e5": frac_s0_gt_1e5,
        "obs_b0_over_S0_hat_median": float(np.median(obs_b0[c] / np.clip(s0_c, 1e-6, None))),
        # DWI RelMSE (G2 shared voxels)
        "WLS_DWI_RelMSE": wls_rel,
        "INR_DWI_RelMSE": inr_rel,
        "DWI_Ratio_INR_over_WLS": float(g["Ratio"]),
        # WLS parameter maps (means on common)
        "WLS_FA_mean": wls_fa_mean,
        "WLS_MD_mean": wls_md_mean,
        "WLS_AD_mean": wls_ad_mean,
        "WLS_RD_mean": wls_rd_mean,
        # INR parameter maps
        "INR_FA_mean": inr_fa_mean,
        "INR_MD_mean": inr_md_mean,
        "INR_AD_mean": inr_ad_mean,
        "INR_RD_mean": inr_rd_mean,
        # Agreement
        "FA_MAE": fa_mae,
        "MD_MAE": md_mae,
        "AD_MAE": ad_mae,
        "RD_MAE": rd_mae,
        "best_loss": float(s.get("best_loss", "nan")),
        "best_epoch": int(float(s.get("best_epoch", -1))),
        "case_label": case,
    }


def main() -> None:
    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    out = exp / "eval_common_mask" / "failure_diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    g2 = _load_g2(exp / "eval_common_mask" / "g2_wls_vs_inr_dwi.csv")
    sm = _load_summary(exp / "eval_common_mask" / "summary.csv")

    rows = []
    for sid in ALL:
        print(f"[diag] {sid}")
        rows.append(diagnose_one(sid, cfg, g2, sm))

    fields = list(rows[0].keys())
    csv_path = out / "failure_3subject_diagnosis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    save_json(out / "failure_3subject_diagnosis.json", {"subjects": rows})

    # Compact readable table
    lines = [
        "# Failure diagnosis: 112920 / 124422 / 130720 (+ control 103515)",
        "",
        "Protocol: `common_mask = brain & WLS_valid`; DWI RelMSE from G2 shared voxels (seed=42, n=131072).",
        "",
        "## Compact table",
        "",
        "| subject | role | N_brain | N_valid | N_common | raw_b0_mean | S0_hat_p99 | S0_max | frac_S0>1e5 | WLS_RelMSE | INR_RelMSE | FA_MAE | MD_MAE | INR_FA_mean | WLS_FA_mean | case |",
        "|---------|------|--------:|--------:|---------:|------------:|-----------:|-------:|------------:|-----------:|-----------:|-------:|-------:|------------:|------------:|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subject']} | {r['role']} | {r['N_brain']} | {r['N_WLS_valid']} | {r['N_common']} | "
            f"{r['raw_b0_mean']:.1f} | {r['S0_hat_p99']:.1f} | {r['S0_hat_max']:.1f} | {r['frac_S0_gt_1e5']:.2e} | "
            f"{r['WLS_DWI_RelMSE']:.4f} | {r['INR_DWI_RelMSE']:.4f} | {r['FA_MAE']:.3f} | {r['MD_MAE']:.6f} | "
            f"{r['INR_FA_mean']:.3f} | {r['WLS_FA_mean']:.3f} | {r['case_label']} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "1. **Raw data hard?** Compare `raw_b0_mean` / `raw_b1000_*` and mask sizes to control. "
        "If similar → not obvious raw-data catastrophe.",
        "2. **WLS itself fails?** Global `WLS_DWI_RelMSE` under `valid_mask` (S0<1e6) is often dominated by rare "
        "`S0_hat` explosions (`frac_S0>1e5`, `S0_max`). High WLS RelMSE ≠ WLS cannot fit typical tissue.",
        "3. **INR truly fails?** Look at `INR_DWI_RelMSE` (~0.12–0.19 vs control ~0.05) and especially "
        "`FA_MAE` (~0.46) with `INR_FA_mean` collapsed vs `WLS_FA_mean`.",
        "",
        "## Verdict template",
        "",
        "- If mask counts & raw intensity ≈ control, but FA_MAE≫0.3 and INR RelMSE elevated → **INR failure**.",
        "- If S0 tails huge and WLS RelMSE huge but FA maps look normal and INR FA collapsed → **INR param failure**; "
        "do not blame raw data solely from WLS RelMSE.",
        "",
    ]
    (out / "failure_3subject_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {out / 'failure_3subject_diagnosis.md'}")
    for r in rows:
        print(
            f"{r['subject']}: WLS={r['WLS_DWI_RelMSE']:.4f} INR={r['INR_DWI_RelMSE']:.4f} "
            f"FA_MAE={r['FA_MAE']:.3f} FA_mean INR/WLS={r['INR_FA_mean']:.3f}/{r['WLS_FA_mean']:.3f} "
            f"S0_max={r['S0_hat_max']:.0f} → {r['case_label']}"
        )


if __name__ == "__main__":
    main()
