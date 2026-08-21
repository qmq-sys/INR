#!/usr/bin/env python
"""
Aggregate Independent INR × N results into mean±std baseline.

Reads outputs/v0_preschema/independent_inr/independent_results.csv
Writes:
  independent_baseline_mean_std.json
  independent_baseline_mean_std.csv
  independent_baseline_mean_std.md
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402

METRICS = [
    "final_loss",
    "FA_mae_vs_wls",
    "MD_mae_vs_wls",
    "AD_mae_vs_wls",
    "RD_mae_vs_wls",
    "dwi_relative_mse",
    "dwi_mse",
]


def _f(x: str | float | None) -> float:
    if x is None or x == "":
        return float("nan")
    return float(x)


def mean_std(vals: list[float]) -> tuple[float, float, int]:
    arr = [v for v in vals if math.isfinite(v)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(arr) / n
    if n == 1:
        return mu, 0.0, 1
    var = sum((v - mu) ** 2 for v in arr) / (n - 1)
    return mu, math.sqrt(var), n


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Independent INR baseline mean±std")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--csv", default="")
    ap.add_argument("--min-epochs", type=int, default=200, help="only keep subjects with epochs>=this")
    ap.add_argument("--exclude", default="", help="comma-separated subject ids to exclude")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = experiment_dir(cfg, "v0_preschema_independent")
    csv_path = Path(args.csv) if args.csv else out_root / "independent_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok", "")).lower() not in {"1", "true", "yes"}:
                continue
            sid = str(r.get("subject_id", "")).strip()
            if sid in exclude:
                continue
            ep = int(float(r.get("epochs") or 0))
            if ep < int(args.min_epochs):
                continue
            rows.append(r)

    summary = {
        "n_subjects": len(rows),
        "min_epochs": int(args.min_epochs),
        "excluded": sorted(exclude),
        "subject_ids": [r["subject_id"] for r in rows],
        "note": (
            "FA/MD/AD/RD MAE are vs conventional WLS-DTI (reference agreement). "
            "Primary physics metric is dwi_relative_mse."
        ),
        "metrics": {},
    }
    for key in METRICS:
        mu, sd, n = mean_std([_f(r.get(key)) for r in rows])
        summary["metrics"][key] = {
            "mean": mu,
            "std": sd,
            "n": n,
            "mean_pm_std": f"{mu:.6g} ± {sd:.6g}",
        }

    save_json(out_root / "independent_baseline_mean_std.json", summary)
    with open(out_root / "independent_baseline_mean_std.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n", "mean_pm_std"])
        for key, v in summary["metrics"].items():
            w.writerow([key, v["mean"], v["std"], v["n"], v["mean_pm_std"]])

    lines = [
        "# Independent INR baseline (mean ± std)",
        "",
        f"- N subjects (epochs ≥ {args.min_epochs}): **{len(rows)}**",
        f"- Note: parameter MAE = difference from WLS reference (not GT error)",
        "",
        "| Metric | mean ± std |",
        "|--------|-----------:|",
    ]
    for key, v in summary["metrics"].items():
        lines.append(f"| {key} | {v['mean_pm_std']} |")
    lines.append("")
    (out_root / "independent_baseline_mean_std.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Baseline] N={len(rows)} from {csv_path}")
    for key, v in summary["metrics"].items():
        print(f"  {key:20s}  {v['mean_pm_std']}")
    print(f"  → {out_root / 'independent_baseline_mean_std.md'}")


if __name__ == "__main__":
    main()
