#!/usr/bin/env python
"""
Experiment 1 — Single-subject INR.

Default outputs (schema-era training, does not overwrite v0):
  outputs/v1_schema_train/single_inr/<sid>/{best.pt, maps.npz, metrics.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.io_utils import experiment_dir, load_config, project_root  # noqa: E402
from inr.metrics_schema import write_summary_and_aggregate  # noqa: E402
from inr.train_independent import resolve_device, train_one_independent_subject  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp1: single-subject INR")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subject", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-voxels", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-traditional-if-exists", action="store_true")
    ap.add_argument("--save-nifti", action="store_true")
    ap.add_argument("--save-tensor", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    e1 = dict(cfg.get("exp1", {}))
    sid = (args.subject or e1.get("subject_id") or "101309").strip()
    epochs = int(args.epochs or e1.get("epochs", 200))
    batch_voxels = int(args.batch_voxels or e1.get("batch_voxels", 4096))

    out_root = experiment_dir(cfg, "v1_schema_train_single")
    out_dir = out_root / sid
    trad_dir = project_root() / cfg.get("output_root", "outputs") / "step1_traditional_dti" / sid

    row = train_one_independent_subject(
        subject_id=sid,
        cfg=cfg,
        out_dir=out_dir,
        trad_dir=trad_dir,
        device=resolve_device(args.device),
        epochs=epochs,
        batch_voxels=batch_voxels,
        lr=float(e1.get("lr", 1e-3)),
        hidden=int(e1.get("hidden", 128)),
        layers=int(e1.get("layers", 4)),
        pe_freqs=int(e1.get("pe_freqs", 8)),
        log_every=int(e1.get("log_every", 10)),
        eval_every=int(e1.get("eval_every", 50)),
        seed=int(e1.get("seed", 42)),
        skip_traditional_if_exists=bool(args.skip_traditional_if_exists),
        save_nifti_flag=bool(args.save_nifti),
        save_tensor_flag=bool(args.save_tensor),
        tag="Exp1",
    )
    write_summary_and_aggregate(out_root, [row])


if __name__ == "__main__":
    main()
