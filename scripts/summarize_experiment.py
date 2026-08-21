#!/usr/bin/env python
"""Rebuild summary.csv + aggregate.csv from an experiment tree of metrics.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inr.io_utils import experiment_dir, load_config  # noqa: E402
from inr.metrics_schema import collect_metrics_from_experiment, write_summary_and_aggregate  # noqa: E402


def main() -> None:
    cfg = load_config()
    default_root = str(experiment_dir(cfg, "v1_schema_reeval_independent"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default=default_root)
    args = ap.parse_args()
    root = Path(args.exp_root)
    rows = collect_metrics_from_experiment(root)
    write_summary_and_aggregate(root, rows)
    print(f"[Summary] N={len(rows)} → {root / 'summary.csv'} + aggregate.csv")


if __name__ == "__main__":
    main()
