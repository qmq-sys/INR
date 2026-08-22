#!/usr/bin/env python
"""Shared INR + Subject Latent — 29-subject MVP training & evaluation.

Does NOT modify Independent INR. One shared network + learnable z_s per subject.

Usage:
  python scripts/train_shared_inr.py
  python scripts/train_shared_inr.py --epochs 200 --device auto
  python scripts/train_shared_inr.py --eval-only
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
from inr.metrics_schema import metrics_json_to_summary_row, write_summary_and_aggregate  # noqa: E402
from inr.train_independent import resolve_device  # noqa: E402
from inr.train_shared import (  # noqa: E402
    build_subject_mapping,
    evaluate_shared_subject,
    load_shared_checkpoint,
    load_subject_mapping,
    prepare_subject_data,
    save_subject_mapping,
    train_shared_inr,
    write_comparison_table,
    write_shared_aggregate_report,
)


def _shared_cfg(cfg: dict) -> dict:
    sc = dict(cfg.get("shared_inr", {}))
    base = dict(cfg.get("exp2", cfg.get("exp1", {})))
    for k in ("epochs", "batch_voxels", "lr", "hidden", "layers", "pe_freqs", "log_every", "seed"):
        if k not in sc and k in base:
            sc[k] = base[k]
    return sc


def _independent_summary_path(cfg: dict) -> Path:
    sc = _shared_cfg(cfg)
    rel = sc.get("independent_summary", "v1_schema_train/independent_inr/eval_common_mask/summary.csv")
    return project_root() / str(cfg.get("output_root", "outputs")) / str(rel)


def run_eval_only(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sc = _shared_cfg(cfg)
    out_root = experiment_dir(cfg, "shared_inr")
    ckpt = out_root / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    device = resolve_device(args.device)
    model, mapping, _ = load_shared_checkpoint(ckpt, device)
    subjects = resolve_subject_list(
        subjects_csv=args.subjects,
        subjects_yaml=args.subjects_yaml,
        subjects_file=project_root() / cfg.get("subjects_file", "subjects_29.txt"),
    )
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    trad_root = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti"
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in subjects:
        if sid not in mapping:
            raise KeyError(f"{sid} not in subject_mapping.json")
        subj = prepare_subject_data(
            subject_id=sid,
            subject_idx=mapping[sid],
            cfg=cfg,
            trad_dir=trad_root / sid,
            skip_traditional_if_exists=True,
        )
        metrics_obj = evaluate_shared_subject(
            model=model,
            subj=subj,
            device=device,
            eval_seed=int(sc.get("seed", 42)),
        )
        save_json = __import__("inr.io_utils", fromlist=["save_json"]).save_json
        save_json(out_root / "metrics" / f"{sid}.json", metrics_obj)
        rows.append(metrics_json_to_summary_row(metrics_obj, ok=True))
        print(f"  [{sid}] FA_MAE={rows[-1]['FA_MAE']:.4f} DWI_RelMSE={rows[-1]['DWI_RelMSE']:.6e}")

    write_summary_and_aggregate(out_root, rows)
    cfg["_independent_summary_path"] = str(_independent_summary_path(cfg))
    write_comparison_table(
        out_root=out_root,
        shared_rows=rows,
        independent_summary=Path(cfg["_independent_summary_path"]),
    )
    write_shared_aggregate_report(out_root, rows, mapping)
    print(f"[SharedINR] eval-only done → {out_root / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared INR + subject latent MVP")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subjects-yaml", default=str(project_root() / "config" / "subjects.yaml"))
    ap.add_argument("--subjects", default="", help="comma-separated override")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-voxels", type=int, default=0)
    ap.add_argument("--latent-dim", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-traditional-if-exists", action="store_true", default=True)
    ap.add_argument("--no-save-maps", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="load best.pt and re-run evaluation only")
    args = ap.parse_args()

    if args.eval_only:
        run_eval_only(args)
        return

    cfg = load_config(args.config)
    sc = _shared_cfg(cfg)
    if not sc.get("enabled", True):
        raise SystemExit("shared_inr.enabled is false in config")

    subjects = resolve_subject_list(
        subjects_csv=args.subjects,
        subjects_yaml=args.subjects_yaml,
        subjects_file=project_root() / cfg.get("subjects_file", "subjects_29.txt"),
    )
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    out_root = experiment_dir(cfg, "shared_inr")
    trad_root = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti"
    device = resolve_device(args.device)

    cfg["_independent_summary_path"] = str(_independent_summary_path(cfg))

    print(f"[SharedINR] N={len(subjects)} latent_dim={int(args.latent_dim or sc.get('latent_dim', 32))}")
    print(f"[SharedINR] out_root={out_root}")
    print(f"[SharedINR] compare vs {_independent_summary_path(cfg)}")

    try:
        train_shared_inr(
            subject_ids=subjects,
            cfg=cfg,
            out_root=out_root,
            trad_root=trad_root,
            device=device,
            latent_dim=int(args.latent_dim or sc.get("latent_dim", 32)),
            epochs=int(args.epochs or sc.get("epochs", 200)),
            batch_voxels=int(args.batch_voxels or sc.get("batch_voxels", 4096)),
            lr=float(sc.get("lr", 1e-3)),
            hidden=int(sc.get("hidden", 128)),
            layers=int(sc.get("layers", 4)),
            pe_freqs=int(sc.get("pe_freqs", 8)),
            log_every=int(sc.get("log_every", 10)),
            seed=int(sc.get("seed", 42)),
            skip_traditional_if_exists=bool(args.skip_traditional_if_exists),
            save_maps=not bool(args.no_save_maps),
            tag="SharedINR",
        )
    except Exception as e:
        print(f"[SharedINR] ERROR: {e}")
        traceback.print_exc()
        err_path = out_root / "error.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps({"error": str(e)}, indent=2) + "\n", encoding="utf-8")
        raise

    print(f"\n[SharedINR] done → {out_root}")


if __name__ == "__main__":
    main()
