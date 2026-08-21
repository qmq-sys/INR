#!/usr/bin/env python
"""
Re-evaluate existing Independent INR checkpoints into the fixed output schema.

Reads:
  outputs/v0_preschema/independent_inr/<sid>/checkpoint.pt

Writes:
  outputs/v1_schema_reeval/independent_inr/<sid>/{best.pt, maps.npz, metrics.json}
  + summary.csv / aggregate.csv

Does NOT retrain — only forward eval under the fixed metrics schema.
"""
from __future__ import annotations

import argparse
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
from inr.io_utils import experiment_dir, load_config, project_root, resolve_subject_list  # noqa: E402
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
    save_subject_outputs,
)


def find_ckpt(legacy_root: Path, new_root: Path, sid: str) -> Path | None:
    for p in (
        new_root / sid / "best.pt",
        legacy_root / sid / "best.pt",
        legacy_root / sid / "checkpoint.pt",
        new_root / sid / "checkpoint.pt",
    ):
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reformat/re-eval Independent INR to fixed schema")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subjects-yaml", default=str(project_root() / "config" / "subjects.yaml"))
    ap.add_argument("--subjects", default="")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-nifti", action="store_true")
    ap.add_argument("--save-tensor", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = resolve_subject_list(
        subjects_csv=args.subjects,
        subjects_yaml=args.subjects_yaml,
        subjects_file=project_root() / cfg.get("subjects_file", "subjects_29.txt"),
    )
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    out_root = experiment_dir(cfg, "v1_schema_reeval_independent")
    legacy_root = experiment_dir(cfg, "v0_preschema_independent")
    trad_root = project_root() / "outputs" / "step1_traditional_dti"
    out_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    rows = []
    for i, sid in enumerate(subjects, start=1):
        print(f"\n===== [{i}/{len(subjects)}] re-eval {sid} =====")
        ckpt_path = find_ckpt(legacy_root, out_root, sid)
        if ckpt_path is None:
            print(f"  SKIP: no checkpoint for {sid}")
            continue

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
        ref = load_or_fit_wls_reference(
            bundle=bundle, trad_dir=trad_root / sid, cfg=cfg, skip_if_exists=True
        )
        coords_np, flat_idx = masked_coords_and_indices(brain)
        coords_t = torch.from_numpy(coords_np)
        bvals_t = torch.from_numpy(bvals_u).to(device)
        bvecs_t = torch.from_numpy(bvecs_u).to(device)

        maps = predict_maps(
            model,
            coords_t,
            flat_idx,
            bundle["data"].shape[:3],
            device,
            want_D=bool(args.save_tensor),
        )
        param_metrics = parameter_agreement_vs_wls(maps, ref, brain & ref["valid_mask"])
        dwi_metrics = evaluate_dwi_reconstruction(
            model,
            coords_t,
            flat_idx,
            dwi.reshape(-1, dwi.shape[-1]),
            bvals_t,
            bvecs_t,
            device,
            seed=42,
        )

        # Prefer training stats from legacy run_meta if present
        legacy_meta = legacy_root / sid / "run_meta.json"
        legacy = json.loads(legacy_meta.read_text(encoding="utf-8")) if legacy_meta.is_file() else {}
        training = {
            "final_loss": legacy.get("final_loss", ckpt.get("best_loss")),
            "best_loss": ckpt.get("best_loss", legacy.get("final_loss")),
            "best_epoch": ckpt.get("best_epoch", legacy.get("epochs")),
            "training_time_sec": legacy.get("sec"),
            "epochs": legacy.get("epochs", ccfg.get("epochs")),
        }
        metrics_obj = build_metrics_json(
            subject_id=sid,
            parameter_metrics=param_metrics,
            dwi=dwi_metrics,
            training=training,
            extra={"source_ckpt": str(ckpt_path)},
        )
        out_dir = out_root / sid
        save_subject_outputs(
            out_dir=out_dir,
            sid=sid,
            model=model,
            maps=maps,
            metrics_obj=metrics_obj,
            ckpt_payload={
                "model": model.state_dict(),
                "subject_id": sid,
                "experiment": "independent_inr",
                "best_epoch": training.get("best_epoch"),
                "best_loss": training.get("best_loss"),
                "config": ccfg,
            },
            affine=bundle["affine"],
            save_nifti_flag=bool(args.save_nifti),
            save_tensor_flag=bool(args.save_tensor),
        )
        row = metrics_json_to_summary_row(metrics_obj, ok=True)
        rows.append(row)
        print(
            f"  FA_MAE={row['FA_MAE']:.4f} MD_MAE={row['MD_MAE']:.6f} "
            f"DWI_RelMSE={row['DWI_RelMSE']:.6e}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summary_and_aggregate(out_root, rows)
    print(f"\n[Reformat] N={len(rows)} → {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
