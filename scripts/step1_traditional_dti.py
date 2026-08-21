#!/usr/bin/env python
"""
Step 1 — Traditional WLS-DTI baseline (no INR).

For each subject:
  DWI (b0+b1000) -> WLS-DTI -> Dxx..Dyz, FA/MD/AD/RD

Writes under outputs/step1_traditional_dti/<sid>/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.dti_fit import fit_dti_b0_b1000  # noqa: E402
from inr.hcp_io import load_hcp_subject  # noqa: E402
from inr.io_utils import load_config, project_root, read_subjects, save_json, save_nifti  # noqa: E402


def process_one(sid: str, hcp_root: str, out_root: Path, cfg: dict) -> dict:
    print(f"\n===== Step1 {sid} =====")
    bundle = load_hcp_subject(hcp_root, sid, b0_threshold=float(cfg["b0_threshold"]))
    fit = fit_dti_b0_b1000(
        bundle["data"],
        bundle["bvals"],
        bundle["bvecs"],
        bundle["brain_mask"],
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
    )
    aff = bundle["affine"]
    sub_dir = out_root / sid
    sub_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(sub_dir / "brain_mask.nii.gz", bundle["brain_mask"].astype("uint8"), aff)
    save_nifti(sub_dir / "valid_mask.nii.gz", fit["valid_mask"].astype("uint8"), aff)
    save_nifti(sub_dir / "S0.nii.gz", fit["S0"], aff)
    save_nifti(sub_dir / "D.nii.gz", fit["D"], aff)
    for k in ("Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz", "FA", "MD", "AD", "RD", "V1"):
        save_nifti(sub_dir / f"{k}.nii.gz", fit[k], aff)

    meta = {
        "subject_id": sid,
        "method": "DIPY_TensorModel_WLS",
        "shells": "b0+b1000",
        "n_volumes_used": fit["n_volumes_used"],
        "n_brain": int(bundle["brain_mask"].sum()),
        "n_valid": int(fit["valid_mask"].sum()),
        "FA_mean": float(fit["FA"][fit["valid_mask"]].mean()) if fit["valid_mask"].any() else None,
        "MD_mean": float(fit["MD"][fit["valid_mask"]].mean()) if fit["valid_mask"].any() else None,
        "diffusion_dir": bundle["diffusion_dir"],
    }
    save_json(sub_dir / "meta.json", meta)
    print(f"[{sid}] volumes={meta['n_volumes_used']} valid={meta['n_valid']} FA_mean={meta['FA_mean']:.4f}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Step1: traditional DTI baseline")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--hcp-root", default="")
    ap.add_argument("--subjects-file", default="")
    ap.add_argument("--subjects", default="", help="comma-separated override")
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hcp_root = args.hcp_root or cfg["hcp_root"]
    if args.subjects.strip():
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects_file = args.subjects_file or str(project_root() / cfg["subjects_file"])
        subjects = read_subjects(subjects_file)
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    out_root = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in subjects:
        try:
            rows.append(process_one(sid, hcp_root, out_root, cfg))
        except Exception as e:
            print(f"[{sid}] ERROR: {e}")
            rows.append({"subject_id": sid, "error": str(e)})

    save_json(out_root / "summary.json", rows)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[Step1] done → {out_root}")


if __name__ == "__main__":
    main()
