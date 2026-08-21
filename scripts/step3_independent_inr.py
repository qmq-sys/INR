#!/usr/bin/env python
"""
Experiment 2 — Independent INR × N subjects.

Each subject gets a fresh SpatialDTIINR (independent θ).

Default layout (schema-era TRAINING, does not overwrite v0 or re-eval):
  outputs/v1_schema_train/independent_inr/
    summary.csv
    aggregate.csv
    <sid>/{best.pt, maps.npz, metrics.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.io_utils import experiment_dir, load_config, project_root, resolve_subject_list  # noqa: E402
from inr.metrics_schema import (  # noqa: E402
    collect_metrics_from_experiment,
    metrics_json_to_summary_row,
    write_summary_and_aggregate,
)
from inr.train_independent import resolve_device, train_one_independent_subject  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp2: Independent INR x N subjects")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subjects-yaml", default=str(project_root() / "config" / "subjects.yaml"))
    ap.add_argument("--subjects", default="", help="comma-separated override")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-voxels", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-done", action="store_true", help="skip if best.pt + metrics.json exist")
    ap.add_argument("--skip-traditional-if-exists", action="store_true", default=True)
    ap.add_argument("--fit-traditional", action="store_true")
    ap.add_argument("--save-nifti", action="store_true")
    ap.add_argument("--save-tensor", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    e2 = dict(cfg.get("exp2", cfg.get("exp1", {})))
    epochs = int(args.epochs or e2.get("epochs", 200))
    batch_voxels = int(args.batch_voxels or e2.get("batch_voxels", 4096))

    subjects = resolve_subject_list(
        subjects_csv=args.subjects,
        subjects_yaml=args.subjects_yaml,
        subjects_file=project_root() / cfg.get("subjects_file", "subjects_29.txt"),
    )
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    out_root = experiment_dir(cfg, "v1_schema_train_independent")
    trad_root = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti"
    out_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    skip_trad = bool(args.skip_traditional_if_exists) and not bool(args.fit_traditional)

    print(f"[Exp2] Independent INR | N={len(subjects)} | epochs={epochs} | device={device}")
    print(f"[Exp2] out_root={out_root}")

    for i, sid in enumerate(subjects, start=1):
        sub_out = out_root / sid
        done = (sub_out / "best.pt").is_file() and (sub_out / "metrics.json").is_file()
        print(f"\n===== [{i}/{len(subjects)}] Independent INR {sid} =====")
        if args.skip_done and done:
            print(f"[Exp2] skip-done: {sub_out}")
            continue
        try:
            train_one_independent_subject(
                subject_id=sid,
                cfg=cfg,
                out_dir=sub_out,
                trad_dir=trad_root / sid,
                device=device,
                epochs=epochs,
                batch_voxels=batch_voxels,
                lr=float(e2.get("lr", 1e-3)),
                hidden=int(e2.get("hidden", 128)),
                layers=int(e2.get("layers", 4)),
                pe_freqs=int(e2.get("pe_freqs", 8)),
                log_every=int(e2.get("log_every", 10)),
                eval_every=int(e2.get("eval_every", 50)),
                seed=int(e2.get("seed", 42)),
                skip_traditional_if_exists=skip_trad,
                save_nifti_flag=bool(args.save_nifti),
                save_tensor_flag=bool(args.save_tensor),
                tag="Exp2",
            )
        except Exception as e:
            print(f"[Exp2] ERROR {sid}: {e}")
            traceback.print_exc()
            sub_out.mkdir(parents=True, exist_ok=True)
            err_metrics = {
                "subject_id": sid,
                "schema_version": 1,
                "parameter_metrics": {},
                "dwi": {"MAE": None, "relative_mse": None},
                "training": {"final_loss": None, "best_loss": None, "best_epoch": None, "training_time_sec": None, "epochs": epochs},
                "n_voxels": None,
            }
            (sub_out / "metrics.json").write_text(
                json.dumps({**err_metrics, "error": str(e)}, indent=2) + "\n",
                encoding="utf-8",
            )

        # refresh master tables after each subject
        rows = []
        for p in sorted(out_root.glob("*/metrics.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            ok = "error" not in obj
            rows.append(metrics_json_to_summary_row(obj, ok=ok, error=str(obj.get("error", ""))))
        write_summary_and_aggregate(out_root, rows)

    rows = collect_metrics_from_experiment(out_root)
    # include failed subjects that have error field
    for p in sorted(out_root.glob("*/metrics.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if "error" in obj:
            # replace/append
            sid = obj.get("subject_id")
            rows = [r for r in rows if r.get("subject_id") != sid]
            rows.append(metrics_json_to_summary_row(obj, ok=False, error=str(obj["error"])))
    rows = sorted(rows, key=lambda r: str(r.get("subject_id")))
    write_summary_and_aggregate(out_root, rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n[Exp2] done: {n_ok}/{len(rows)} → {out_root / 'summary.csv'} + aggregate.csv")


if __name__ == "__main__":
    main()
