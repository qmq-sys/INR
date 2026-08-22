"""Experiment 3 analysis: Independent vs Shared across sampling levels."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics_schema import _stats, write_summary_and_aggregate
from .volume_sampling import DEFAULT_LEVELS, pct_dir_name

FOCUS_SUBJECTS = ("112920", "124422", "130720", "101309")
METRIC_KEYS = ("DWI_RelMSE", "FA_MAE", "MD_MAE", "AD_MAE", "RD_MAE")


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_level_summaries(
    exp_root: Path,
    level_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d = exp_root / pct_dir_name(level_label)
    ind = _read_summary(d / "independent_inr" / "summary.csv")
    if not ind:
        ind = _read_summary(d / "independent" / "summary.csv")
    shared = _read_summary(d / "shared_inr" / "summary.csv")
    if not shared:
        shared = _read_summary(d / "shared" / "summary.csv")
    return ind, shared


def build_level_comparison(
    *,
    level_label: str,
    sampling_fraction: float,
    independent_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
    n_dwi_volumes: int | None = None,
) -> list[dict[str, Any]]:
    ind = {r["subject_id"]: r for r in independent_rows}
    sh = {r["subject_id"]: r for r in shared_rows}
    sids = sorted(set(ind) & set(sh))
    out: list[dict[str, Any]] = []
    for sid in sids:
        ir, sr = ind[sid], sh[sid]
        row: dict[str, Any] = {
            "subject_id": sid,
            "sampling_level": level_label,
            "sampling_fraction": sampling_fraction,
            "n_dwi_volumes": n_dwi_volumes or sr.get("n_volumes") or ir.get("n_volumes"),
            "Independent_DWI_RelMSE": _f(ir.get("DWI_RelMSE")),
            "Shared_DWI_RelMSE": _f(sr.get("DWI_RelMSE")),
            "DWI_delta": _f(sr.get("DWI_RelMSE")) - _f(ir.get("DWI_RelMSE")),
            "Independent_FA_MAE": _f(ir.get("FA_MAE")),
            "Shared_FA_MAE": _f(sr.get("FA_MAE")),
            "FA_delta": _f(sr.get("FA_MAE")) - _f(ir.get("FA_MAE")),
            "Independent_MD_MAE": _f(ir.get("MD_MAE")),
            "Shared_MD_MAE": _f(sr.get("MD_MAE")),
            "MD_delta": _f(sr.get("MD_MAE")) - _f(ir.get("MD_MAE")),
            "Independent_AD_MAE": _f(ir.get("AD_MAE")),
            "Shared_AD_MAE": _f(sr.get("AD_MAE")),
            "AD_delta": _f(sr.get("AD_MAE")) - _f(ir.get("AD_MAE")),
            "Independent_RD_MAE": _f(ir.get("RD_MAE")),
            "Shared_RD_MAE": _f(sr.get("RD_MAE")),
            "RD_delta": _f(sr.get("RD_MAE")) - _f(ir.get("RD_MAE")),
            "Shared_better_DWI": int(_f(sr.get("DWI_RelMSE")) < _f(ir.get("DWI_RelMSE"))),
            "Shared_better_FA": int(_f(sr.get("FA_MAE")) < _f(ir.get("FA_MAE"))),
        }
        out.append(row)
    return out


def write_experiment3_tables(
    exp_root: Path,
    protocol: dict[str, Any],
    *,
    independent_100_path: Path,
    shared_100_path: Path,
) -> dict[str, Any]:
    exp_root = Path(exp_root)
    all_cmp: list[dict[str, Any]] = []
    cohort: dict[str, list[dict[str, Any]]] = {}

    for label, frac in DEFAULT_LEVELS:
        level_dir = exp_root / pct_dir_name(label)
        level_dir.mkdir(parents=True, exist_ok=True)
        n_vol = int(protocol["levels"][label]["n_total"])

        if label == "100%":
            ind_rows = _read_summary(independent_100_path)
            if shared_100_path.suffix == ".json":
                sh_obj = json.loads(shared_100_path.read_text(encoding="utf-8"))
                shared_rows = sh_obj.get("rows") or sh_obj
                if isinstance(shared_rows, dict):
                    shared_rows = sh_obj.get("rows", [])
            else:
                shared_rows = _read_summary(shared_100_path)
            # materialize copies for experiment tree
            ind_out = level_dir / "independent_inr"
            sh_out = level_dir / "shared_inr"
            ind_out.mkdir(parents=True, exist_ok=True)
            sh_out.mkdir(parents=True, exist_ok=True)
            write_summary_and_aggregate(ind_out, ind_rows)
            write_summary_and_aggregate(sh_out, shared_rows)
        else:
            ind_rows, shared_rows = load_level_summaries(exp_root, label)

        cmp_rows = build_level_comparison(
            level_label=label,
            sampling_fraction=float(frac),
            independent_rows=ind_rows,
            shared_rows=shared_rows,
            n_dwi_volumes=n_vol,
        )
        cmp_path = level_dir / "comparison_independent_vs_shared.csv"
        with open(cmp_path, "w", newline="", encoding="utf-8") as f:
            if cmp_rows:
                w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
                w.writeheader()
                w.writerows(cmp_rows)
        all_cmp.extend(cmp_rows)
        cohort[label] = cmp_rows

    master = exp_root / "experiment3_all_comparisons.csv"
    if all_cmp:
        with open(master, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_cmp[0].keys()))
            w.writeheader()
            w.writerows(all_cmp)

    agg = _cohort_aggregate(cohort, protocol)
    (exp_root / "aggregate.md").write_text(agg["markdown"], encoding="utf-8")
    return {"comparison_rows": all_cmp, "aggregate": agg}


def _cohort_aggregate(cohort: dict[str, list[dict]], protocol: dict[str, Any]) -> dict[str, Any]:
    lines = [
        "# Experiment 3 — Low DWI Sampling (Independent vs Shared)",
        "",
        f"- Protocol seed: **{protocol.get('seed')}**",
        f"- Reference subject for volume indices: **{protocol.get('reference_subject')}**",
        "- Training: nested volume subsets (b0 + b1000); spatial training mask = brain",
        "- Evaluation: common_mask = brain & WLS_valid; eval seed=42; max_voxels=131072",
        "- 100% results reused from frozen Independent + Shared MVP (no retrain)",
        "",
        "## Cohort medians",
        "",
        "| level | n_vol | Independent DWI | Shared DWI | Δ DWI | Independent FA | Shared FA | Δ FA | Shared better DWI | Shared better FA |",
        "|-------|------:|----------------:|-----------:|------:|---------------:|----------:|-----:|------------------:|-----------------:|",
    ]
    summary_stats: dict[str, Any] = {}
    for label, _ in DEFAULT_LEVELS:
        rows = cohort.get(label, [])
        if not rows:
            continue
        idwi = [_f(r["Independent_DWI_RelMSE"]) for r in rows]
        sdwi = [_f(r["Shared_DWI_RelMSE"]) for r in rows]
        ifa = [_f(r["Independent_FA_MAE"]) for r in rows]
        sfa = [_f(r["Shared_FA_MAE"]) for r in rows]
        ddwi = [_f(r["DWI_delta"]) for r in rows]
        dfa = [_f(r["FA_delta"]) for r in rows]
        n_vol = protocol["levels"][label]["n_total"]
        sb_dwi = sum(int(r["Shared_better_DWI"]) for r in rows)
        sb_fa = sum(int(r["Shared_better_FA"]) for r in rows)
        lines.append(
            f"| {label} | {n_vol} | {np.median(idwi):.4g} | {np.median(sdwi):.4g} | {np.median(ddwi):+.4g} | "
            f"{np.median(ifa):.4g} | {np.median(sfa):.4g} | {np.median(dfa):+.4g} | {sb_dwi}/{len(rows)} | {sb_fa}/{len(rows)} |"
        )
        summary_stats[label] = {
            "median_Independent_DWI": float(np.median(idwi)),
            "median_Shared_DWI": float(np.median(sdwi)),
            "median_DWI_delta": float(np.median(ddwi)),
            "median_Independent_FA": float(np.median(ifa)),
            "median_Shared_FA": float(np.median(sfa)),
            "median_FA_delta": float(np.median(dfa)),
            "n_shared_better_dwi": sb_dwi,
            "n_shared_better_fa": sb_fa,
            "n_subjects": len(rows),
        }

    lines += ["", "## Focus subjects", ""]
    for sid in FOCUS_SUBJECTS:
        lines.append(f"### {sid}")
        lines.append("")
        lines.append("| level | Ind DWI | Sh DWI | Δ DWI | Ind FA | Sh FA | Δ FA |")
        lines.append("|-------|--------:|-------:|------:|-------:|------:|-----:|")
        for label, _ in DEFAULT_LEVELS:
            rows = cohort.get(label, [])
            r = next((x for x in rows if x["subject_id"] == sid), None)
            if not r:
                continue
            lines.append(
                f"| {label} | {r['Independent_DWI_RelMSE']:.4g} | {r['Shared_DWI_RelMSE']:.4g} | {r['DWI_delta']:+.4g} | "
                f"{r['Independent_FA_MAE']:.4g} | {r['Shared_FA_MAE']:.4g} | {r['FA_delta']:+.4g} |"
            )
        lines.append("")

    lines += [
        "## Interpretation (candidate only)",
        "",
        "- Look for **crossover**: Shared curve degrades slower as sampling drops.",
        "- Negative Δ DWI / Δ FA means Shared better than Independent at that subject/level.",
        "- No automatic claim of superiority — inspect curves in `figures/`.",
        "",
    ]
    return {"markdown": "\n".join(lines), "stats": summary_stats}


def plot_experiment3_figures(exp_root: Path, protocol: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    exp_root = Path(exp_root)
    fig_dir = exp_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    master = exp_root / "experiment3_all_comparisons.csv"
    if not master.is_file():
        return
    with open(master, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    order = [lbl for lbl, _ in DEFAULT_LEVELS]
    fracs = [float(protocol["levels"][lbl]["fraction"]) * 100 for lbl in order]

    def _median_by_level(key: str, model: str) -> list[float]:
        out = []
        for lbl in order:
            sub = [r for r in rows if r["sampling_level"] == lbl]
            vals = [_f(r[f"{model}_{key}"]) for r in sub]
            out.append(float(np.median(vals)) if vals else float("nan"))
        return out

    # Figure 1: DWI RelMSE vs sampling
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, _median_by_level("DWI_RelMSE", "Independent"), "o-", label="Independent", lw=2)
    ax.plot(fracs, _median_by_level("DWI_RelMSE", "Shared"), "s-", label="Shared", lw=2)
    ax.set_xlabel("Sampling (%)")
    ax.set_ylabel("DWI RelMSE (median)")
    ax.set_title("Experiment 3: DWI RelMSE vs sampling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "dwi_relmse_vs_sampling.png", dpi=150)
    plt.close(fig)

    # Figure 2: FA MAE vs sampling
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, _median_by_level("FA_MAE", "Independent"), "o-", label="Independent", lw=2)
    ax.plot(fracs, _median_by_level("FA_MAE", "Shared"), "s-", label="Shared", lw=2)
    ax.set_xlabel("Sampling (%)")
    ax.set_ylabel("FA MAE vs WLS (median)")
    ax.set_title("Experiment 3: FA agreement vs sampling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fa_mae_vs_sampling.png", dpi=150)
    plt.close(fig)

    # Figure 3: subject-wise DWI delta heatmap (29 x 4)
    sids = sorted({r["subject_id"] for r in rows})
    mat = np.full((len(sids), len(order)), np.nan)
    for j, lbl in enumerate(order):
        for i, sid in enumerate(sids):
            r = next((x for x in rows if x["subject_id"] == sid and x["sampling_level"] == lbl), None)
            if r:
                mat[i, j] = _f(r["DWI_delta"])
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=6)
    for i, sid in enumerate(sids):
        if sid in FOCUS_SUBJECTS:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#c0392b")
    ax.set_title("Δ DWI RelMSE (Shared − Independent)\nblue = Shared better")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "subject_dwi_delta_heatmap.png", dpi=150)
    plt.close(fig)
