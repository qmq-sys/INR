#!/usr/bin/env python
"""Rebuild independent_results.csv from per-subject run_meta.json files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402

FIELDS = [
    "subject_id",
    "ok",
    "error",
    "final_loss",
    "FA_mae_vs_wls",
    "MD_mae_vs_wls",
    "AD_mae_vs_wls",
    "RD_mae_vs_wls",
    "dwi_relative_mse",
    "dwi_mse",
    "epochs",
    "n_brain_voxels",
    "n_volumes",
    "sec",
    "out_dir",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = experiment_dir(cfg, "v0_preschema_independent")
    rows = []
    for meta in sorted(root.glob("*/run_meta.json")):
        obj = json.loads(meta.read_text(encoding="utf-8"))
        rows.append(obj)
    save_json(root / "independent_results.json", rows)
    with open(root / "independent_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print(f"[Rebuild] {len(rows)} subjects → {root / 'independent_results.csv'}")


if __name__ == "__main__":
    main()
