#!/usr/bin/env python
"""Phase 1: re-evaluate existing Independent INR best.pt under common DWI mask.

Does NOT retrain. Does NOT change best.pt semantics (still min training-loss).

Outputs (under v1_schema_train/independent_inr/diagnostics/):
  29_subject_summary.csv
  29_subject_summary.json
  common_mask_report.md
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
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.common_dwi_eval import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    DEFAULT_MAX_VOXELS,
    evaluate_wls_inr_dwi_common,
)
from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402
from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import resolve_device  # noqa: E402

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit(f"nibabel required: {e}") from e

FAILURES = ("130720", "112920", "124422")

SUMMARY_FIELDS = [
    "subject",
    "WLS_DWI_RelMSE_common",
    "INR_DWI_RelMSE_common",
    "WLS_DWI_MAE_common",
    "INR_DWI_MAE_common",
    "legacy_INR_DWI_RelMSE",
    "best_loss",
    "best_loss_epoch",
    "best_DWI_RelMSE",
    "best_DWI_epoch",
    "final_DWI_RelMSE",
    "FA_MAE",
    "MD_MAE",
    "AD_MAE",
    "RD_MAE",
    "n_common_dwi_voxels",
    "n_eval_dwi_voxels",
    "n_common_dwi_values",
    "n_wls_nonfinite",
    "n_inr_nonfinite",
    "eval_seed",
    "max_voxels",
    "note_best_DWI",
]


def _nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _wls_D(trad: Path) -> np.ndarray:
    D = _nii(trad / "D.nii.gz").astype(np.float32)
    if D.ndim == 5 and D.shape[-2:] == (3, 3):
        return D
    if D.ndim == 4 and D.shape[-1] == 9:
        return D.reshape(*D.shape[:3], 3, 3)
    raise ValueError(f"unexpected D shape {D.shape} in {trad}")


def _load_summary(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["subject_id"]: r for r in csv.DictReader(f)}


def _load_inr(ckpt_path: Path, device: torch.device) -> SpatialDTIINR:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ccfg = ckpt.get("config", {})
    model = SpatialDTIINR(
        hidden=int(ccfg.get("hidden", 128)),
        layers=int(ccfg.get("layers", 4)),
        pe_freqs=int(ccfg.get("pe_freqs", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _median(vals: list[Any]) -> float:
    arr = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "N/A"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(x) else f"{x:.6g}"


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fail_rows = [r for r in rows if r["subject"] in FAILURES]
    lines = [
        "# Common-mask DWI re-evaluation (Phase 1)",
        "",
        "## Common mask definition (model-independent)",
        "",
        "```text",
        "common_dwi_eval_mask =",
        "    brain_mask",
        "    AND WLS_valid_mask   # from inr/dti_fit.py valid_mask (unchanged)",
        "    AND finite(observed_DWI)   # all volumes finite at the voxel",
        "```",
        "",
        "- Prediction finiteness (`finite(WLS_pred)` / `finite(INR_pred)`) is **not** in the common mask.",
        "- If a prediction is nonfinite on the shared eval set, RelMSE is set to **NaN** (no silent voxel drop).",
        "- WLS and INR use the **same** deterministic `eval_voxel_indices` (seed=42, cap=131072).",
        "- `best.pt` still means **minimum training-loss** checkpoint; this phase does not retrain.",
        "- `best_DWI_epoch` / `best_DWI_RelMSE` are **not recoverable** from existing runs → recorded as empty / N/A.",
        "- No `1 < S0 < 15000` filter is used.",
        "",
        "## Aggregate",
        "",
        f"- N subjects: {len(rows)}",
        f"- WLS RelMSE_common median: {_median([r['WLS_DWI_RelMSE_common'] for r in rows]):.6g}",
        f"- INR RelMSE_common median: {_median([r['INR_DWI_RelMSE_common'] for r in rows]):.6g}",
        f"- Total WLS nonfinite entries (all subjects): {sum(int(r['n_wls_nonfinite']) for r in rows)}",
        f"- Total INR nonfinite entries (all subjects): {sum(int(r['n_inr_nonfinite']) for r in rows)}",
        "",
        "## Per-subject table",
        "",
        "| subject | n_common | n_eval | WLS RelMSE | INR RelMSE | legacy INR RelMSE | n_wls_nf | n_inr_nf | FA_MAE |",
        "|---------|---------:|-------:|-----------:|-----------:|------------------:|---------:|---------:|-------:|",
    ]
    for r in sorted(rows, key=lambda z: -float(z["FA_MAE"])):
        mark = " **" if r["subject"] in FAILURES else ""
        lines.append(
            f"| {r['subject']}{mark} | {int(r['n_common_dwi_voxels'])} | {int(r['n_eval_dwi_voxels'])} | "
            f"{float(r['WLS_DWI_RelMSE_common']):.6g} | {float(r['INR_DWI_RelMSE_common']):.6g} | "
            f"{_fmt(r.get('legacy_INR_DWI_RelMSE'))} | {int(r['n_wls_nonfinite'])} | "
            f"{int(r['n_inr_nonfinite'])} | {float(r['FA_MAE']):.4f} |"
        )
    lines += ["", "## Known failure subjects", ""]
    for r in fail_rows:
        lines += [
            f"### {r['subject']}",
            "",
            f"- n_common_dwi_voxels: {r['n_common_dwi_voxels']}",
            f"- n_eval_dwi_voxels: {r['n_eval_dwi_voxels']}",
            f"- n_common_dwi_values: {r['n_common_dwi_values']}",
            f"- WLS_DWI_RelMSE_common: {r['WLS_DWI_RelMSE_common']}",
            f"- INR_DWI_RelMSE_common: {r['INR_DWI_RelMSE_common']}",
            f"- legacy_INR_DWI_RelMSE (brain-only protocol): {r.get('legacy_INR_DWI_RelMSE')}",
            f"- n_wls_nonfinite: {r['n_wls_nonfinite']}",
            f"- n_inr_nonfinite: {r['n_inr_nonfinite']}",
            f"- FA_MAE (unchanged vs WLS maps): {r['FA_MAE']}",
            f"- best_loss / best_loss_epoch: {r['best_loss']} / {r['best_loss_epoch']}",
            f"- best_DWI_*: {r['note_best_DWI']}",
            "",
        ]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 common-mask DWI re-eval of existing best.pt")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--max-voxels", type=int, default=DEFAULT_MAX_VOXELS)
    ap.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    args = ap.parse_args()

    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    trad_root = project_root() / "outputs" / "step1_traditional_dti"
    out = exp / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    legacy = _load_summary(exp / "summary.csv")
    sids = list(legacy.keys())
    if int(args.max_subjects) > 0:
        sids = sids[: int(args.max_subjects)]

    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(sids, 1):
        print(f"[{i}/{len(sids)}] common-mask re-eval {sid}")
        rec = legacy[sid]
        trad = trad_root / sid
        ckpt_path = exp / sid / "best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
        bvals = bundle["bvals"]
        bvecs = normalize_bvecs(bvals, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
        vol_m = shell_volume_mask(
            bvals,
            b0_threshold=float(cfg["b0_threshold"]),
            shell_tol=float(cfg["shell_tol"]),
            shells=tuple(cfg.get("dti_shells", [1000.0])),
            include_b0=True,
        )
        dwi = np.ascontiguousarray(bundle["data"][..., vol_m], dtype=np.float32)
        brain = np.ascontiguousarray(bundle["brain_mask"])
        valid = np.ascontiguousarray(_nii(trad / "valid_mask.nii.gz") > 0.5)
        S0 = np.ascontiguousarray(_nii(trad / "S0.nii.gz"), dtype=np.float32)
        D = np.ascontiguousarray(_wls_D(trad))

        model = _load_inr(ckpt_path, device)
        ev = evaluate_wls_inr_dwi_common(
            observed_dwi=dwi,
            brain_mask=brain,
            wls_valid_mask=valid,
            wls_S0=S0,
            wls_D=D,
            model=model,
            bvals=bvals[vol_m],
            bvecs=bvecs[vol_m],
            device=device,
            max_voxels=int(args.max_voxels),
            seed=int(args.seed),
        )
        ev.pop("flat_idx", None)

        row = {
            "subject": sid,
            "WLS_DWI_RelMSE_common": ev["WLS_DWI_RelMSE_common"],
            "INR_DWI_RelMSE_common": ev["INR_DWI_RelMSE_common"],
            "WLS_DWI_MAE_common": ev["WLS_DWI_MAE_common"],
            "INR_DWI_MAE_common": ev["INR_DWI_MAE_common"],
            "legacy_INR_DWI_RelMSE": float(rec["DWI_RelMSE"]),
            "best_loss": float(rec["best_loss"]),
            "best_loss_epoch": int(float(rec["best_epoch"])),
            "best_DWI_RelMSE": "",
            "best_DWI_epoch": "",
            "final_DWI_RelMSE": "",
            "FA_MAE": float(rec["FA_MAE"]),
            "MD_MAE": float(rec["MD_MAE"]),
            "AD_MAE": float(rec["AD_MAE"]),
            "RD_MAE": float(rec["RD_MAE"]),
            "n_common_dwi_voxels": ev["n_common_dwi_voxels"],
            "n_eval_dwi_voxels": ev["n_eval_dwi_voxels"],
            "n_common_dwi_values": ev["n_common_dwi_values"],
            "n_wls_nonfinite": ev["n_wls_nonfinite"],
            "n_inr_nonfinite": ev["n_inr_nonfinite"],
            "eval_seed": ev["eval_seed"],
            "max_voxels": ev["max_voxels"],
            "note_best_DWI": "N/A for existing checkpoints (eval_every unused; not logged historically)",
        }
        rows.append(row)
        print(
            f"    common={row['n_common_dwi_voxels']} eval={row['n_eval_dwi_voxels']} "
            f"WLS={row['WLS_DWI_RelMSE_common']:.6g} INR={row['INR_DWI_RelMSE_common']:.6g} "
            f"legacy_INR={row['legacy_INR_DWI_RelMSE']:.6g} "
            f"nf(WLS/INR)={row['n_wls_nonfinite']}/{row['n_inr_nonfinite']}"
        )

        del model, bundle, dwi
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out / "29_subject_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    payload = {
        "phase": 1,
        "experiment": "v1_schema_train_independent",
        "common_mask_definition": "brain AND WLS_valid AND finite(observed_DWI)",
        "note": (
            "Primary WLS/INR DWI comparison uses identical voxel indices under the common mask. "
            "legacy_INR_DWI_RelMSE is the previous brain-only protocol. "
            "best_DWI_* unavailable for historical best.pt runs."
        ),
        "n_subjects": len(rows),
        "subjects": rows,
    }
    save_json(out / "29_subject_summary.json", payload)
    _write_report(out / "common_mask_report.md", rows)
    print(f"\n[phase1] wrote {csv_path}")
    print(f"[phase1] wrote {out / '29_subject_summary.json'}")
    print(f"[phase1] wrote {out / 'common_mask_report.md'}")


if __name__ == "__main__":
    main()
