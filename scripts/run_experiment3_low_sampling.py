#!/usr/bin/env python
"""Experiment 3: Low DWI-volume sampling — Independent vs Shared INR.

Does NOT change architecture / loss / physics / eval protocol.
100% reuses frozen Independent + Shared MVP results (no retrain).

Usage:
  python scripts/run_experiment3_low_sampling.py --summarize-only
  python scripts/run_experiment3_low_sampling.py --train-levels 50,25,10
  python scripts/run_experiment3_low_sampling.py --train-levels 50 --max-subjects 2  # smoke
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

from inr.experiment3_analysis import (  # noqa: E402
    plot_experiment3_figures,
    write_experiment3_tables,
)
from inr.io_utils import experiment_dir, load_config, project_root, resolve_subject_list  # noqa: E402
from inr.metrics_schema import metrics_json_to_summary_row, write_summary_and_aggregate  # noqa: E402
from inr.train_independent import resolve_device, train_one_independent_subject  # noqa: E402
from inr.train_shared import train_shared_inr  # noqa: E402
from inr.volume_sampling import (  # noqa: E402
    DEFAULT_LEVELS,
    level_fraction,
    level_indices,
    load_sampling_protocol,
    pct_dir_name,
    protocol_from_config_reference,
    save_sampling_protocol,
)


def _exp3_root(cfg: dict) -> Path:
    rel = (cfg.get("experiments") or {}).get("experiment3_low_sampling", "experiment3_low_sampling")
    return project_root() / str(cfg.get("output_root", "outputs")) / str(rel)


def _independent_100_summary(cfg: dict) -> Path:
    sc = cfg.get("experiment3", {}) or {}
    rel = sc.get(
        "independent_summary_100",
        "v1_schema_train/independent_inr/eval_common_mask/summary.csv",
    )
    return project_root() / str(cfg.get("output_root", "outputs")) / str(rel)


def _shared_100_summary(cfg: dict) -> Path:
    sc = cfg.get("experiment3", {}) or {}
    rel = sc.get("shared_summary_100", "shared_inr/summary.json")
    p = project_root() / str(cfg.get("output_root", "outputs")) / str(rel)
    if p.is_file():
        return p
    return project_root() / str(cfg.get("output_root", "outputs")) / "shared_inr" / "summary.csv"


def _train_cfg(cfg: dict) -> dict:
    base = dict(cfg.get("exp2", cfg.get("exp1", {})))
    e3 = dict(cfg.get("experiment3", {}) or {})
    out = dict(base)
    for k in ("epochs", "batch_voxels", "lr", "hidden", "layers", "pe_freqs", "log_every", "seed"):
        if k in e3:
            out[k] = e3[k]
    return out


def _parse_levels(s: str) -> list[str]:
    if not s.strip():
        return ["50%", "25%", "10%"]
    mapping = {"100": "100%", "50": "50%", "25": "25%", "10": "10%"}
    out = []
    for part in s.split(","):
        p = part.strip().rstrip("%")
        lbl = mapping.get(p, f"{p}%")
        out.append(lbl)
    return out


def train_independent_level(
    *,
    level_label: str,
    subject_ids: list[str],
    cfg: dict,
    protocol: dict,
    exp_root: Path,
    device,
    tc: dict,
    skip_trad: bool,
) -> None:
    vi = level_indices(protocol, level_label)
    frac = level_fraction(protocol, level_label)
    out_root = exp_root / pct_dir_name(level_label) / "independent_inr"
    trad_root = project_root() / str(cfg.get("output_root", "outputs")) / "step1_traditional_dti"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sid in enumerate(subject_ids, 1):
        print(f"\n[Exp3-Ind {level_label}] [{i}/{len(subject_ids)}] {sid}")
        sub_out = out_root / sid
        try:
            row = train_one_independent_subject(
                subject_id=sid,
                cfg=cfg,
                out_dir=sub_out,
                trad_dir=trad_root / sid,
                device=device,
                epochs=int(tc["epochs"]),
                batch_voxels=int(tc["batch_voxels"]),
                lr=float(tc["lr"]),
                hidden=int(tc["hidden"]),
                layers=int(tc["layers"]),
                pe_freqs=int(tc["pe_freqs"]),
                log_every=int(tc.get("log_every", 10)),
                eval_every=int(tc.get("eval_every", 50)),
                seed=int(tc["seed"]),
                skip_traditional_if_exists=skip_trad,
                tag=f"Exp3-Ind-{level_label}",
                train_volume_indices=vi,
                sampling_fraction=frac,
            )
            rows.append(row)
        except Exception as e:
            print(f"[Exp3-Ind] ERROR {sid}: {e}")
            traceback.print_exc()
            rows.append(metrics_json_to_summary_row({"subject_id": sid, "dwi": {}, "parameter_metrics": {}, "training": {}}, ok=False, error=str(e)))

    write_summary_and_aggregate(out_root, rows)


def train_shared_level(
    *,
    level_label: str,
    subject_ids: list[str],
    cfg: dict,
    protocol: dict,
    exp_root: Path,
    device,
    tc: dict,
    scfg: dict,
    skip_trad: bool,
) -> None:
    vi = level_indices(protocol, level_label)
    frac = level_fraction(protocol, level_label)
    out_root = exp_root / pct_dir_name(level_label) / "shared_inr"
    trad_root = project_root() / str(cfg.get("output_root", "outputs")) / "step1_traditional_dti"
    cfg = dict(cfg)
    cfg["_independent_summary_path"] = str(_independent_100_summary(cfg))

    train_shared_inr(
        subject_ids=subject_ids,
        cfg=cfg,
        out_root=out_root,
        trad_root=trad_root,
        device=device,
        latent_dim=int(scfg.get("latent_dim", 32)),
        epochs=int(tc["epochs"]),
        batch_voxels=int(tc["batch_voxels"]),
        lr=float(tc["lr"]),
        hidden=int(tc["hidden"]),
        layers=int(tc["layers"]),
        pe_freqs=int(tc["pe_freqs"]),
        log_every=int(tc.get("log_every", 10)),
        seed=int(tc["seed"]),
        skip_traditional_if_exists=skip_trad,
        save_maps=False,
        tag=f"Exp3-Shared-{level_label}",
        train_volume_indices=vi,
        sampling_fraction=frac,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 3: low DWI sampling")
    ap.add_argument("--config", default=str(project_root() / "config" / "default.yaml"))
    ap.add_argument("--subjects-yaml", default=str(project_root() / "config" / "subjects.yaml"))
    ap.add_argument("--subjects", default="")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--train-levels", default="50,25,10", help="comma list: 50,25,10 (100% reused)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--skip-independent", action="store_true")
    ap.add_argument("--skip-shared", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reference-subject", default="")
    ap.add_argument("--epochs", type=int, default=0, help="override training epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_root = _exp3_root(cfg)
    exp_root.mkdir(parents=True, exist_ok=True)
    proto_path = exp_root / "sampling_protocol.json"

    ref_sid = args.reference_subject.strip() or str((cfg.get("experiment3") or {}).get("reference_subject", "101309"))
    seed = int((cfg.get("experiment3") or {}).get("seed", 42))

    if proto_path.is_file():
        protocol = load_sampling_protocol(proto_path)
        print(f"[Exp3] loaded {proto_path}")
    else:
        protocol = protocol_from_config_reference(cfg, reference_subject=ref_sid, seed=seed)
        save_sampling_protocol(proto_path, protocol)
        print(f"[Exp3] wrote {proto_path}")
        for lbl, _ in DEFAULT_LEVELS:
            lv = protocol["levels"][lbl]
            print(f"  {lbl}: n_total={lv['n_total']} (b0={lv['n_b0']}, b1000={lv['n_b1000']})")

    if args.summarize_only:
        write_experiment3_tables(
            exp_root,
            protocol,
            independent_100_path=_independent_100_summary(cfg),
            shared_100_path=_shared_100_summary(cfg),
        )
        plot_experiment3_figures(exp_root, protocol)
        print(f"[Exp3] summary + figures → {exp_root}")
        return

    levels = _parse_levels(args.train_levels)
    subjects = resolve_subject_list(
        subjects_csv=args.subjects,
        subjects_yaml=args.subjects_yaml,
        subjects_file=project_root() / cfg.get("subjects_file", "subjects_29.txt"),
    )
    if int(args.max_subjects) > 0:
        subjects = subjects[: int(args.max_subjects)]

    tc = _train_cfg(cfg)
    if int(args.epochs) > 0:
        tc["epochs"] = int(args.epochs)
    scfg = dict(cfg.get("shared_inr", {}))
    device = resolve_device(args.device)
    skip_trad = True

    print(f"[Exp3] train levels={levels} N={len(subjects)} device={device}")

    for level in levels:
        if level == "100%":
            print(f"[Exp3] skip train at 100% (reuse existing)")
            continue
        if level not in protocol["levels"]:
            raise KeyError(f"unknown level {level}")

        if not args.skip_independent:
            train_independent_level(
                level_label=level,
                subject_ids=subjects,
                cfg=cfg,
                protocol=protocol,
                exp_root=exp_root,
                device=device,
                tc=tc,
                skip_trad=skip_trad,
            )
        if not args.skip_shared:
            train_shared_level(
                level_label=level,
                subject_ids=subjects,
                cfg=cfg,
                protocol=protocol,
                exp_root=exp_root,
                device=device,
                tc=tc,
                scfg=scfg,
                skip_trad=skip_trad,
            )

    write_experiment3_tables(
        exp_root,
        protocol,
        independent_100_path=_independent_100_summary(cfg),
        shared_100_path=_shared_100_summary(cfg),
    )
    plot_experiment3_figures(exp_root, protocol)
    print(f"\n[Exp3] done → {exp_root}")


if __name__ == "__main__":
    main()
