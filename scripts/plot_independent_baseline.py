#!/usr/bin/env python
"""Four scatter/distribution figures for Independent INR × 29."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inr.io_utils import experiment_dir, load_config  # noqa: E402

INK = "#1f2420"
MUTED = "#6b716c"
GRID = "#e6e4de"
FA_C = "#2f5d50"
MD_C = "#8c4a2f"
AD_C = "#355f7a"
RD_C = "#6a5a2c"
DWI_C = "#3c3a38"
OUT_C = "#9b2c2c"
OUTLIERS = {"112920", "124422", "130720"}


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _ids(rows: list[dict[str, str]]) -> list[str]:
    return [str(r["subject_id"]) for r in rows]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Cambria", "Times New Roman"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.8,
        }
    )


def _split(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    ids = _ids(rows)
    vals = _col(rows, key)
    out_m = np.array([i in OUTLIERS for i in ids])
    return vals[~out_m], vals[out_m], [i for i, m in zip(ids, out_m) if not m], [i for i, m in zip(ids, out_m) if m]


def _strip_hist(ax, vals_ok: np.ndarray, vals_out: np.ndarray, color: str, xlabel: str) -> None:
    all_v = np.concatenate([vals_ok, vals_out]) if len(vals_out) else vals_ok
    bins = np.histogram_bin_edges(all_v, bins="sturges")
    ax.hist(all_v, bins=bins, color=color, alpha=0.28, edgecolor=color, linewidth=0.7, zorder=1)
    med = float(np.median(all_v))
    ax.axvline(med, color=INK, lw=1.05, ls="--", alpha=0.8, zorder=2)
    ymin = ax.get_ylim()[0]
    ax.plot(vals_ok, np.full_like(vals_ok, ymin), "|", color=color, markersize=10, markeredgewidth=1.1, zorder=3, clip_on=False)
    if len(vals_out):
        ax.plot(vals_out, np.full_like(vals_out, ymin), "D", color=OUT_C, markersize=5, zorder=4, clip_on=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("subjects")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ymax = ax.get_ylim()[1]
    ax.text(med, ymax * 0.92, f"median {med:.3g}", ha="center", va="top", fontsize=8, color=MUTED)


def _scatter_r_vs_mae(ax, rows, mae_key, r_key, color, xlabel, ylabel) -> None:
    ids = _ids(rows)
    x = _col(rows, r_key)
    y = _col(rows, mae_key)
    ok = np.array([i not in OUTLIERS for i in ids])
    ax.scatter(x[ok], y[ok], s=36, c=color, edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(x[~ok], y[~ok], s=52, c=OUT_C, marker="D", edgecolors="white", linewidths=0.5, zorder=4)
    for sid, xi, yi in zip(ids, x, y):
        if sid in OUTLIERS:
            ax.annotate(sid, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7, color=OUT_C)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _finish(fig, path: Path, title: str) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="regular")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def plot_fa(rows, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ok, out, _, _ = _split(rows, "FA_MAE")
    _strip_hist(axes[0], ok, out, FA_C, "FA MAE vs WLS")
    _scatter_r_vs_mae(axes[1], rows, "FA_MAE", "FA_r", FA_C, "FA Pearson r vs WLS", "FA MAE")
    axes[0].set_title("Distribution")
    axes[1].set_title("Agreement vs error")
    _finish(fig, out_dir / "fig1_fa.png", "Independent INR × 29  —  FA")


def plot_md(rows, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ok, out, _, _ = _split(rows, "MD_MAE")
    _strip_hist(axes[0], ok, out, MD_C, r"MD MAE vs WLS  (mm$^2$/s)")
    _scatter_r_vs_mae(axes[1], rows, "MD_MAE", "MD_r", MD_C, "MD Pearson r vs WLS", r"MD MAE  (mm$^2$/s)")
    axes[0].set_title("Distribution")
    axes[1].set_title("Agreement vs error")
    _finish(fig, out_dir / "fig2_md.png", "Independent INR × 29  —  MD")


def plot_ad_rd(rows, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ids = _ids(rows)
    ad = _col(rows, "AD_MAE")
    rd = _col(rows, "RD_MAE")
    out_m = np.array([i in OUTLIERS for i in ids])
    rng = np.random.default_rng(1)
    axes[0].scatter(
        np.full(np.sum(~out_m), 0) + rng.uniform(-0.08, 0.08, np.sum(~out_m)),
        ad[~out_m],
        s=28,
        c=AD_C,
        edgecolors="white",
        linewidths=0.4,
        label="AD",
        zorder=3,
    )
    axes[0].scatter(
        np.full(np.sum(~out_m), 1) + rng.uniform(-0.08, 0.08, np.sum(~out_m)),
        rd[~out_m],
        s=28,
        c=RD_C,
        edgecolors="white",
        linewidths=0.4,
        label="RD",
        zorder=3,
    )
    if np.any(out_m):
        axes[0].scatter(np.full(np.sum(out_m), 0), ad[out_m], s=42, c=OUT_C, marker="D", zorder=4)
        axes[0].scatter(np.full(np.sum(out_m), 1), rd[out_m], s=42, c=OUT_C, marker="D", zorder=4)
    axes[0].set_xticks([0, 1], ["AD MAE", "RD MAE"])
    axes[0].set_ylabel(r"MAE vs WLS  (mm$^2$/s)")
    axes[0].set_title("Distribution")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_axisbelow(True)
    axes[0].set_xlim(-0.5, 1.5)

    ok = ~out_m
    axes[1].scatter(ad[ok], rd[ok], s=36, c=AD_C, edgecolors="white", linewidths=0.5, zorder=3)
    axes[1].scatter(ad[~ok], rd[~ok], s=52, c=OUT_C, marker="D", edgecolors="white", linewidths=0.5, zorder=4)
    for sid, x, y in zip(ids, ad, rd):
        if sid in OUTLIERS:
            axes[1].annotate(sid, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=OUT_C)
    lim = max(float(ad.max()), float(rd.max())) * 1.08
    axes[1].plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=":")
    axes[1].set_xlabel(r"AD MAE  (mm$^2$/s)")
    axes[1].set_ylabel(r"RD MAE  (mm$^2$/s)")
    axes[1].set_title("AD vs RD")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(color=GRID, lw=0.6)
    axes[1].set_axisbelow(True)
    _finish(fig, out_dir / "fig3_ad_rd.png", "Independent INR × 29  —  AD / RD")


def plot_dwi(rows, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ok, out, _, _ = _split(rows, "DWI_RelMSE")
    _strip_hist(axes[0], ok, out, DWI_C, "DWI relative MSE")
    ids = _ids(rows)
    x = _col(rows, "DWI_RelMSE")
    y = _col(rows, "FA_MAE")
    out_m = np.array([i in OUTLIERS for i in ids])
    axes[1].scatter(x[~out_m], y[~out_m], s=36, c=DWI_C, edgecolors="white", linewidths=0.5, zorder=3)
    axes[1].scatter(x[out_m], y[out_m], s=52, c=OUT_C, marker="D", edgecolors="white", linewidths=0.5, zorder=4)
    for sid, xi, yi in zip(ids, x, y):
        if sid in OUTLIERS:
            axes[1].annotate(sid, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7, color=OUT_C)
    axes[1].set_xlabel("DWI relative MSE")
    axes[1].set_ylabel("FA MAE vs WLS")
    axes[0].set_title("Distribution")
    axes[1].set_title("Signal vs FA agreement")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(color=GRID, lw=0.6)
    axes[1].set_axisbelow(True)
    _finish(fig, out_dir / "fig4_dwi.png", "Independent INR × 29  —  DWI reconstruction")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    csv_path = Path(args.csv) if args.csv else exp / "summary.csv"
    out_dir = Path(args.out_dir) if args.out_dir else exp / "figures"
    rows = _load_rows(csv_path)
    _style()
    plot_fa(rows, out_dir)
    plot_md(rows, out_dir)
    plot_ad_rd(rows, out_dir)
    plot_dwi(rows, out_dir)


if __name__ == "__main__":
    main()
