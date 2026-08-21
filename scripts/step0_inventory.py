#!/usr/bin/env python
"""
Step 0 — Data inventory for all subjects.

Writes:
  outputs/step0_inventory/inventory.csv
  outputs/step0_inventory/inventory.json
  outputs/step0_inventory/<sid>_shells.txt
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

from inr.hcp_io import inventory_hcp_subject  # noqa: E402
from inr.io_utils import count_shells, load_config, project_root, read_subjects, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Step0: inventory HCP subjects")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--hcp-root", default="")
    ap.add_argument("--subjects-file", default="")
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hcp_root = args.hcp_root or cfg["hcp_root"]
    subjects_file = args.subjects_file or str(project_root() / cfg["subjects_file"])
    subjects = read_subjects(subjects_file)
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    out_dir = project_root() / cfg.get("output_root", "outputs") / "step0_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in subjects:
        print(f"[Step0] {sid}")
        try:
            bundle = inventory_hcp_subject(hcp_root, sid, b0_threshold=float(cfg["b0_threshold"]))
            shells = count_shells(
                bundle["bvals"],
                b0_threshold=float(cfg["b0_threshold"]),
                shell_tol=float(cfg["shell_tol"]),
            )
            shape = bundle["shape"]
            row = {
                "subject_id": sid,
                "ok": True,
                "error": "",
                "diffusion_dir": bundle["diffusion_dir"],
                "dwi_path": bundle["dwi_path"],
                "shape_xyz": f"{shape[0]}x{shape[1]}x{shape[2]}",
                "n_volumes": int(shape[3]),
                "n_brain_voxels": int(bundle["n_brain_voxels"]),
                **{k: shells[k] for k in ("n_b0", "n_b1000", "n_b2000", "n_b3000", "n_diffusion_dirs", "shell_type", "is_multi_shell")},
                "unique_bvals": ",".join(str(x) for x in shells["unique_bvals"]),
            }
            txt = (
                f"Subject {sid}:\n"
                f"  b=0     : {shells['n_b0']}\n"
                f"  b=1000  : {shells['n_b1000']}\n"
                f"  b=2000  : {shells['n_b2000']}\n"
                f"  b=3000  : {shells['n_b3000']}\n"
                f"  dirs    : {shells['n_diffusion_dirs']}\n"
                f"  type    : {shells['shell_type']}\n"
                f"  shape   : {row['shape_xyz']} x {row['n_volumes']}\n"
            )
            (out_dir / f"{sid}_shells.txt").write_text(txt, encoding="utf-8")
            print(txt)
        except Exception as e:
            row = {
                "subject_id": sid,
                "ok": False,
                "error": str(e),
                "diffusion_dir": "",
                "dwi_path": "",
                "shape_xyz": "",
                "n_volumes": 0,
                "n_brain_voxels": 0,
                "n_b0": 0,
                "n_b1000": 0,
                "n_b2000": 0,
                "n_b3000": 0,
                "n_diffusion_dirs": 0,
                "shell_type": "error",
                "is_multi_shell": False,
                "unique_bvals": "",
            }
            print(f"  ERROR: {e}")
        rows.append(row)

    save_json(out_dir / "inventory.json", rows)
    csv_path = out_dir / "inventory.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n[Step0] done: {n_ok}/{len(rows)} ok → {out_dir}")


if __name__ == "__main__":
    main()
