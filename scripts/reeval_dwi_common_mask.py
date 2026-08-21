#!/usr/bin/env python
"""Re-evaluate existing Independent INR best.pt under common_mask = brain & WLS_valid.

Does NOT retrain. Does NOT change best.pt weights/semantics.
Writes:
  outputs/v1_schema_train/independent_inr/eval_common_mask/
    summary.csv
    aggregate.md
    <sid>/metrics.json
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.coords import masked_coords_and_indices  # noqa: E402
from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402
from inr.metrics_schema import (  # noqa: E402
    build_metrics_json,
    metrics_json_to_summary_row,
    parameter_agreement_vs_wls,
    write_summary_and_aggregate,
)
from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import (  # noqa: E402
    evaluate_dwi_reconstruction,
    load_or_fit_wls_reference,
    predict_maps,
    resolve_device,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-eval best.pt with DWI on brain & WLS_valid")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--subject", default="", help="optional single subject id")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    out_root = exp / "eval_common_mask"
    out_root.mkdir(parents=True, exist_ok=True)
    trad_root = project_root() / "outputs" / "step1_traditional_dti"
    device = resolve_device(args.device)

    legacy_summary = exp / "summary.csv"
    old_dwi: dict[str, float] = {}
    if legacy_summary.is_file():
        with open(legacy_summary, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                old_dwi[r["subject_id"]] = float(r["DWI_RelMSE"])

    if args.subject.strip():
        sids = [args.subject.strip()]
    else:
        sids = sorted(
            p.parent.name
            for p in exp.glob("*/best.pt")
            if p.parent.name.isdigit() or p.parent.name.replace("_", "").isalnum()
        )
        # Prefer summary order if available
        if legacy_summary.is_file():
            with open(legacy_summary, newline="", encoding="utf-8") as f:
                ordered = [r["subject_id"] for r in csv.DictReader(f)]
            sids = [s for s in ordered if (exp / s / "best.pt").is_file()]
    if int(args.max_subjects) > 0:
        sids = sids[: int(args.max_subjects)]

    rows = []
    for i, sid in enumerate(sids, 1):
        print(f"\n===== [{i}/{len(sids)}] common-mask eval {sid} =====")
        ckpt_path = exp / sid / "best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

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
        ref = load_or_fit_wls_reference(
            bundle=bundle, trad_dir=trad_root / sid, cfg=cfg, skip_if_exists=True
        )

        common_mask = brain & ref["valid_mask"]
        n_brain = int(np.count_nonzero(brain))
        n_wls_valid = int(np.count_nonzero(ref["valid_mask"]))
        n_common = int(np.count_nonzero(common_mask))
        print(f"[EvalMask] brain: {n_brain}  WLS_valid: {n_wls_valid}  common: {n_common}")
        if n_common == 0:
            raise RuntimeError(f"{sid}: empty common_mask — stop; do not change thresholds")
        if n_common > n_brain or not np.all(~common_mask | brain):
            raise RuntimeError(f"{sid}: common_mask not ⊆ brain")

        train_coords_np, train_flat_idx = masked_coords_and_indices(brain)
        train_coords_t = torch.from_numpy(train_coords_np)
        eval_coords_np, eval_flat_idx = masked_coords_and_indices(common_mask)
        eval_coords_t = torch.from_numpy(eval_coords_np)
        bvals_t = torch.from_numpy(bvals[vol_m].astype(np.float32)).to(device)
        bvecs_t = torch.from_numpy(bvecs[vol_m].astype(np.float32)).to(device)
        dwi_flat = dwi.reshape(-1, dwi.shape[-1])

        maps = predict_maps(
            model, train_coords_t, train_flat_idx, bundle["data"].shape[:3], device, want_D=False
        )
        param_metrics = parameter_agreement_vs_wls(maps, ref, common_mask)
        dwi_metrics = evaluate_dwi_reconstruction(
            model,
            eval_coords_t,
            eval_flat_idx,
            dwi_flat,
            bvals_t,
            bvecs_t,
            device,
            max_voxels=131072,
            seed=42,
            evaluation_mask="brain & WLS_valid",
        )

        # Carry training stats from existing metrics.json if present
        old_mj = exp / sid / "metrics.json"
        training = {
            "final_loss": ckpt.get("best_loss"),
            "best_loss": ckpt.get("best_loss"),
            "best_epoch": ckpt.get("best_epoch"),
            "training_time_sec": None,
            "epochs": ccfg.get("epochs"),
        }
        if old_mj.is_file():
            prev = json.loads(old_mj.read_text(encoding="utf-8"))
            training.update(prev.get("training", {}))

        metrics_obj = build_metrics_json(
            subject_id=sid,
            parameter_metrics=param_metrics,
            dwi=dwi_metrics,
            training=training,
            extra={
                "source_ckpt": str(ckpt_path),
                "training_mask": "brain",
                "evaluation_mask": "brain & WLS_valid",
                "n_brain_voxels": n_brain,
                "n_wls_valid_voxels": n_wls_valid,
                "n_common_voxels": n_common,
                "legacy_brain_only_DWI_RelMSE": old_dwi.get(sid),
            },
        )
        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "metrics.json", metrics_obj)
        row = metrics_json_to_summary_row(metrics_obj, ok=True)
        row["legacy_DWI_RelMSE"] = old_dwi.get(sid)
        rows.append(row)

        old_v = old_dwi.get(sid)
        old_s = f"{old_v:.6e}" if old_v is not None else "N/A"
        print(
            f"  FA_MAE={row['FA_MAE']:.4f} DWI_RelMSE={row['DWI_RelMSE']:.6e} "
            f"(legacy brain-only={old_s}) "
            f"n_eval={int(dwi_metrics['n_eval_voxels'])} n_sampled={int(dwi_metrics['n_sampled_voxels'])}"
        )

        del model, bundle, dwi, maps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary_and_aggregate(out_root, rows)
    # Also a compact comparison table
    cmp_path = out_root / "dwi_relmse_brain_vs_common.csv"
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "legacy_brain_DWI_RelMSE",
                "common_DWI_RelMSE",
                "DWI_MAE",
                "FA_MAE",
                "MD_MAE",
                "AD_MAE",
                "RD_MAE",
                "n_eval_voxels",
                "n_sampled_voxels",
                "evaluation_mask",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "subject_id": r["subject_id"],
                    "legacy_brain_DWI_RelMSE": r.get("legacy_DWI_RelMSE"),
                    "common_DWI_RelMSE": r["DWI_RelMSE"],
                    "DWI_MAE": r["DWI_MAE"],
                    "FA_MAE": r["FA_MAE"],
                    "MD_MAE": r["MD_MAE"],
                    "AD_MAE": r["AD_MAE"],
                    "RD_MAE": r["RD_MAE"],
                    "n_eval_voxels": r.get("n_eval_voxels"),
                    "n_sampled_voxels": r.get("n_sampled_voxels"),
                    "evaluation_mask": r.get("evaluation_mask"),
                }
            )
    print(f"\n[eval] wrote {out_root / 'summary.csv'}")
    print(f"[eval] wrote {cmp_path}")


if __name__ == "__main__":
    main()
