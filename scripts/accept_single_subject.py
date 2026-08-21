#!/usr/bin/env python
"""
Accept / re-evaluate a trained Independent INR checkpoint.

Reports FA/MD/AD/RD vs WLS (reference agreement) + DWI reconstruction.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json, save_nifti  # noqa: E402
from inr.metrics_schema import parameter_agreement_vs_wls  # noqa: E402
from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import (  # noqa: E402
    evaluate_dwi_reconstruction,
    load_or_fit_wls_reference,
    predict_maps,
    resolve_device,
)
from inr.coords import masked_coords_and_indices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Accept single-subject INR (FA/MD/AD/RD + DWI)")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subject", default="101309")
    ap.add_argument(
        "--ckpt",
        default="",
        help="checkpoint.pt path (default: step2 or step3 for subject)",
    )
    ap.add_argument("--source", choices=["step2", "step3", "v0-single", "v0-independent", "v1-reeval", "auto"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sid = args.subject.strip()
    v0_single = experiment_dir(cfg, "v0_preschema_single") / sid
    v0_ind = experiment_dir(cfg, "v0_preschema_independent") / sid
    v1_reeval = experiment_dir(cfg, "v1_schema_reeval_independent") / sid
    v1_train_ind = experiment_dir(cfg, "v1_schema_train_independent") / sid
    v1_train_single = experiment_dir(cfg, "v1_schema_train_single") / sid

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        save_dir = ckpt_path.parent
    else:
        src = args.source
        if src in ("step2", "v0-single"):
            ckpt_path = v0_single / "checkpoint.pt"
            save_dir = v0_single
        elif src in ("step3", "v0-independent"):
            ckpt_path = v0_ind / "checkpoint.pt"
            save_dir = v0_ind
        elif src == "v1-reeval":
            ckpt_path = v1_reeval / "best.pt"
            save_dir = v1_reeval
        else:
            candidates = [
                (v1_train_ind / "best.pt", v1_train_ind),
                (v1_train_single / "best.pt", v1_train_single),
                (v1_reeval / "best.pt", v1_reeval),
                (v0_ind / "checkpoint.pt", v0_ind),
                (v0_single / "checkpoint.pt", v0_single),
            ]
            ckpt_path, save_dir = next(((p, d) for p, d in candidates if p.is_file()), (v0_ind / "checkpoint.pt", v0_ind))

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = resolve_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ccfg = ckpt.get("config", {})
    model = SpatialDTIINR(
        hidden=int(ccfg.get("hidden", 128)),
        layers=int(ccfg.get("layers", 4)),
        pe_freqs=int(ccfg.get("pe_freqs", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
    bvals = bundle["bvals"]
    bvecs = normalize_bvecs(bvals, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
    brain = bundle["brain_mask"]
    vol_m = shell_volume_mask(
        bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    dwi = bundle["data"][..., vol_m].astype(np.float32)
    bvals_u = bvals[vol_m].astype(np.float32)
    bvecs_u = bvecs[vol_m].astype(np.float32)

    trad_dir = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti" / sid
    ref = load_or_fit_wls_reference(
        bundle=bundle, trad_dir=trad_dir, cfg=cfg, skip_if_exists=True
    )
    coords_np, flat_idx = masked_coords_and_indices(brain)
    coords_t = torch.from_numpy(coords_np)
    bvals_t = torch.from_numpy(bvals_u).to(device)
    bvecs_t = torch.from_numpy(bvecs_u).to(device)
    dwi_flat = dwi.reshape(-1, dwi.shape[-1])

    maps = predict_maps(
        model, coords_t, flat_idx, bundle["data"].shape[:3], device, want_D=bool(args.save_maps)
    )
    param_metrics = parameter_agreement_vs_wls(maps, ref, brain & ref["valid_mask"])
    dwi_metrics = evaluate_dwi_reconstruction(
        model,
        coords_t,
        flat_idx,
        dwi_flat,
        bvals_t,
        bvecs_t,
        device,
        max_voxels=131072,
        seed=42,
    )

    report = {
        "subject_id": sid,
        "checkpoint": str(ckpt_path),
        "note": "FA/MD/AD/RD MAE are vs conventional WLS-DTI (reference agreement, not GT error)",
        "parameter_vs_wls": {
            "FA_mae": param_metrics["FA"]["MAE"],
            "MD_mae": param_metrics["MD"]["MAE"],
            "AD_mae": param_metrics["AD"]["MAE"],
            "RD_mae": param_metrics["RD"]["MAE"],
            "FA_rmse": param_metrics["FA"]["RMSE"],
            "MD_rmse": param_metrics["MD"]["RMSE"],
            "AD_rmse": param_metrics["AD"]["RMSE"],
            "RD_rmse": param_metrics["RD"]["RMSE"],
            "n_voxels": param_metrics["n_voxels"],
        },
        "dwi_reconstruction": dwi_metrics,
        "map_stats_in_valid_mask": {},
    }
    m = brain & ref["valid_mask"]
    for k in ("FA", "MD", "AD", "RD"):
        v = maps[k][m]
        r = ref[k][m]
        report["map_stats_in_valid_mask"][k] = {
            "inr_mean": float(v.mean()),
            "inr_std": float(v.std()),
            "wls_mean": float(r.mean()),
            "wls_std": float(r.std()),
        }

    if args.save_maps:
        for k in ("S0", "FA", "MD", "AD", "RD"):
            save_nifti(save_dir / f"{k}_inr.nii.gz", maps[k], bundle["affine"])

    out_json = save_dir / "acceptance_report.json"
    save_json(out_json, report)
    print(f"[Accept] subject={sid}")
    print(f"  ckpt: {ckpt_path}")
    p = report["parameter_vs_wls"]
    print(
        f"  vs WLS  FA_mae={p['FA_mae']:.4f}  MD_mae={p['MD_mae']:.6f}  "
        f"AD_mae={p['AD_mae']:.6f}  RD_mae={p['RD_mae']:.6f}"
    )
    d = report["dwi_reconstruction"]
    print(f"  DWI     relative_mse={d['relative_mse']:.6e}  mae={d['MAE']:.6e}")
    print(f"  saved → {out_json}")


if __name__ == "__main__":
    main()
