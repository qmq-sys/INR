#!/usr/bin/env python
"""Gate G2: WLS vs INR DWI RelMSE on identical common-mask voxels.

common_mask = brain & WLS_valid
Same deterministic sample (seed=42, max_voxels=131072) for both methods.

Writes:
  outputs/v1_schema_train/independent_inr/eval_common_mask/g2_wls_vs_inr_dwi.csv
  .../g2_wls_vs_inr_dwi.md
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.common_dwi_eval import evaluate_wls_inr_dwi_common  # noqa: E402
from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402
from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import resolve_device  # noqa: E402

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit(f"nibabel required: {e}") from e

FIELDS = [
    "Subject",
    "WLS_DWI_RelMSE",
    "INR_DWI_RelMSE",
    "Delta",
    "Ratio",
    "n_eval_voxels",
    "n_sampled_voxels",
    "n_wls_nonfinite",
    "n_inr_nonfinite",
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


def _load_inr(ckpt: Path, device: torch.device) -> SpatialDTIINR:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="G2: paired WLS/INR DWI RelMSE")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--subject", default="")
    args = ap.parse_args()

    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    out = exp / "eval_common_mask"
    out.mkdir(parents=True, exist_ok=True)
    trad_root = project_root() / "outputs" / "step1_traditional_dti"
    device = resolve_device("auto")

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

    rows = []
    for i, sid in enumerate(sids, 1):
        print(f"[{i}/{len(sids)}] G2 {sid}")
        ckpt = exp / sid / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        trad = trad_root / sid
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
        model = _load_inr(ckpt, device)

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
            max_voxels=131072,
            seed=42,
        )
        ev.pop("flat_idx", None)
        ev.pop("shape_xyz", None)

        row = {
            "Subject": sid,
            "WLS_DWI_RelMSE": ev["WLS_DWI_RelMSE"],
            "INR_DWI_RelMSE": ev["INR_DWI_RelMSE"],
            "Delta": ev["Delta"],
            "Ratio": ev["Ratio"],
            "n_eval_voxels": ev["n_eval_dwi_voxels"],
            "n_sampled_voxels": ev["n_sampled_voxels"],
            "n_wls_nonfinite": ev["n_wls_nonfinite"],
            "n_inr_nonfinite": ev["n_inr_nonfinite"],
        }
        rows.append(row)
        print(
            f"    WLS={row['WLS_DWI_RelMSE']:.6g}  INR={row['INR_DWI_RelMSE']:.6g}  "
            f"Δ={row['Delta']:.6g}  ratio={row['Ratio']:.4g}  "
            f"n_samp={row['n_sampled_voxels']}"
        )
        del model, bundle, dwi
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out / "g2_wls_vs_inr_dwi.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    save_json(
        out / "g2_wls_vs_inr_dwi.json",
        {
            "gate": "G2",
            "common_mask": "brain & WLS_valid",
            "sampling": {"seed": 42, "max_voxels": 131072, "shared_indices": True},
            "n_subjects": len(rows),
            "rows": rows,
        },
    )

    lines = [
        "# G2 — WLS vs INR DWI RelMSE (shared common-mask voxels)",
        "",
        "- common_mask = `brain & WLS_valid`",
        "- identical deterministic sample: seed=42, max_voxels=131072",
        "- Delta = INR − WLS; Ratio = INR / WLS",
        "",
        "| Subject | WLS_DWI_RelMSE | INR_DWI_RelMSE | Delta | Ratio | n_sampled |",
        "|---------|---------------:|---------------:|------:|------:|----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Subject']} | {float(r['WLS_DWI_RelMSE']):.6g} | {float(r['INR_DWI_RelMSE']):.6g} | "
            f"{float(r['Delta']):.6g} | {float(r['Ratio']):.4g} | {int(r['n_sampled_voxels'])} |"
        )
    lines.append("")
    (out / "g2_wls_vs_inr_dwi.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[G2] wrote {csv_path}")
    print(f"[G2] wrote {out / 'g2_wls_vs_inr_dwi.md'}")


if __name__ == "__main__":
    main()
