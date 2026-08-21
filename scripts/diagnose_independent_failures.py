#!/usr/bin/env python
"""Diagnose Independent INR failures vs WLS signal reconstruction.

Does not retrain. Writes:
  outputs/v1_schema_train/independent_inr/diagnostics/
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.io_utils import experiment_dir, load_config, project_root, save_json  # noqa: E402
from inr.metrics_schema import dwi_reconstruction_metrics  # noqa: E402
from inr.model import SpatialDTIINR  # noqa: E402
from inr.physics import dti_forward_signal  # noqa: E402
from inr.train_independent import resolve_device  # noqa: E402

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit(f"nibabel required: {e}") from e

FAILURES = ("130720", "112920", "124422")
CONTROL = "103515"
FA_FAIL = 0.30
FA_MOD = 0.18
DWI_FAIL = 0.12
DWI_MOD = 0.09
WLS_BAD = 0.08
INK = "#1f2420"
GRID = "#e6e4de"
MAX_VOX = 131072
SEED = 42
S0_SANE_MAX = 15000.0
S0_SANE_MIN = 1.0


def _load_summary(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _wls_D(trad: Path) -> np.ndarray:
    D = _nii(trad / "D.nii.gz").astype(np.float32)
    if D.ndim == 5 and D.shape[-2:] == (3, 3):
        return D
    if D.ndim == 4 and D.shape[-1] == 9:
        return D.reshape(*D.shape[:3], 3, 3)
    raise ValueError(f"unexpected D shape {D.shape} in {trad}")


def _sample_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= MAX_VOX:
        return np.arange(n)
    return rng.choice(n, size=MAX_VOX, replace=False)


@torch.no_grad()
def wls_dwi_relmse(
    *,
    S0: np.ndarray,
    D: np.ndarray,
    dwi: np.ndarray,
    mask: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Same RelMSE formula / seed / voxel cap as Independent INR eval."""
    flat = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if flat.size == 0:
        return {"relative_mse": float("nan"), "MAE": float("nan"), "n_values": 0, "n_voxels_sampled": 0}
    rng = np.random.default_rng(SEED)
    take = flat[_sample_idx(int(flat.size), rng)]
    S0_v = torch.from_numpy(np.ascontiguousarray(S0.reshape(-1)[take], dtype=np.float32)).to(device)
    D_v = torch.from_numpy(np.ascontiguousarray(D.reshape(-1, 3, 3)[take], dtype=np.float32)).to(device)
    obs = torch.from_numpy(np.ascontiguousarray(dwi.reshape(-1, dwi.shape[-1])[take], dtype=np.float32)).to(device)
    bvals_t = torch.from_numpy(np.asarray(bvals, dtype=np.float32)).to(device)
    bvecs_t = torch.from_numpy(np.asarray(bvecs, dtype=np.float32)).to(device)
    pred = dti_forward_signal(S0_v, D_v, bvals_t, bvecs_t)
    out = dwi_reconstruction_metrics(pred.cpu().numpy(), obs.cpu().numpy())
    out["n_voxels_sampled"] = int(take.size)
    return out


def _status_fa(x: float) -> str:
    if x >= FA_FAIL:
        return "failure"
    if x >= FA_MOD:
        return "moderate"
    return "stable"


def _status_dwi(x: float) -> str:
    if x >= DWI_FAIL:
        return "failure"
    if x >= DWI_MOD:
        return "moderate"
    return "stable"


def _overall(fa_s: str, dwi_s: str) -> str:
    if "failure" in (fa_s, dwi_s):
        return "failure"
    if "moderate" in (fa_s, dwi_s):
        return "moderate"
    return "stable"


def _case(wls: float, inr: float, fa: float) -> str:
    wls_bad = wls >= WLS_BAD
    inr_bad = inr >= DWI_FAIL
    fa_bad = fa >= FA_FAIL
    if wls_bad:
        return "A_data_or_wls"
    if inr_bad:
        return "B_inr_reconstruction"
    if fa_bad:
        return "C_parameter_ambiguity"
    return "ok"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Cambria", "Times New Roman"],
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_wls_vs_inr(rows: list[dict], out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    x = np.array([float(r["WLS_DWI_RelMSE"]) for r in rows])
    y = np.array([float(r["INR_DWI_RelMSE"]) for r in rows])
    ids = [r["subject_id"] for r in rows]
    fail = np.array([i in FAILURES for i in ids])
    lim = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.12
    ax.plot([0, lim], [0, lim], color="#8a8680", lw=0.8, ls=":")
    ax.scatter(x[~fail], y[~fail], s=32, c="#2f5d50", edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(x[fail], y[fail], s=54, c="#9b2c2c", marker="D", edgecolors="white", linewidths=0.4, zorder=4)
    for sid, xi, yi in zip(ids, x, y):
        if sid in FAILURES:
            ax.annotate(sid, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#9b2c2c")
    ax.set_xlabel("WLS DWI RelMSE")
    ax.set_ylabel("INR DWI RelMSE")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("WLS vs Independent INR  —  DWI reconstruction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_training_curves(v0_root: Path, v1_root: Path, out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {CONTROL: "#2f5d50", "130720": "#9b2c2c", "112920": "#8c4a2f", "124422": "#355f7a"}
    for sid in (CONTROL, *FAILURES):
        p = v0_root / sid / "loss.csv"
        if not p.is_file():
            continue
        ep, loss = [], []
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ep.append(int(r["epoch"]))
                loss.append(float(r["loss"]))
        ax.plot(ep, loss, color=colors[sid], lw=1.2, label=sid)
        mj = v1_root / sid / "metrics.json"
        if mj.is_file():
            best = int(json.loads(mj.read_text(encoding="utf-8"))["training"]["best_epoch"])
            ax.axvline(best, color=colors[sid], lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("DWI relative training loss")
    ax.set_title("Training curves  (v0 log; dashed = v1 best_epoch)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _mid_slice(mask: np.ndarray) -> int:
    return int(np.argmax(np.asarray(mask).sum(axis=(0, 1))))


def _show(ax, img, title, vmin=None, vmax=None, cmap="gray") -> None:
    ax.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_param_maps(sid: str, trad: Path, maps_npz: Path, out: Path) -> None:
    fa_w = _nii(trad / "FA.nii.gz")
    md_w = _nii(trad / "MD.nii.gz")
    ad_w = _nii(trad / "AD.nii.gz")
    rd_w = _nii(trad / "RD.nii.gz")
    mask = _nii(trad / "valid_mask.nii.gz") > 0.5
    npz = np.load(maps_npz)
    fa_i, md_i, ad_i, rd_i = npz["FA"], npz["MD"], npz["AD"], npz["RD"]
    sl = _mid_slice(mask)
    vis = mask[:, :, sl]
    _style()
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 10.2))
    rows = [
        ("FA", fa_w[:, :, sl] * vis, fa_i[:, :, sl] * vis, 0.0, 1.0),
        ("MD", md_w[:, :, sl] * vis, md_i[:, :, sl] * vis, 0.0, 0.002),
        ("AD", ad_w[:, :, sl] * vis, ad_i[:, :, sl] * vis, 0.0, 0.0025),
        ("RD", rd_w[:, :, sl] * vis, rd_i[:, :, sl] * vis, 0.0, 0.002),
    ]
    for r, (name, w, i, vmin, vmax) in enumerate(rows):
        _show(axes[r, 0], w, f"WLS {name}", vmin, vmax, "gray")
        _show(axes[r, 1], i, f"INR {name}", vmin, vmax, "gray")
        d = np.abs(w - i)
        vmax_d = max(vmax * 0.6, float(np.percentile(d[vis], 95) if vis.any() else vmax))
        _show(axes[r, 2], d, f"|{name} diff|", 0.0, vmax_d, "magma")
    fig.suptitle(f"{sid}  parameter maps  (z={sl})", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_dwi_and_evals(
    *,
    sid: str,
    cfg: dict,
    trad: Path,
    ckpt_path: Path,
    maps_npz: Path,
    device: torch.device,
    out_dwi: Path,
    out_eig: Path,
) -> None:
    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
    bvals = bundle["bvals"]
    bvecs = normalize_bvecs(bvals, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
    vol_m = shell_volume_mask(
        bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    dwi = bundle["data"][..., vol_m].astype(np.float32)
    bvals_u = bvals[vol_m].astype(np.float32)
    bvecs_u = bvecs[vol_m].astype(np.float32)
    brain = bundle["brain_mask"]
    valid = _nii(trad / "valid_mask.nii.gz") > 0.5
    S0_w = _nii(trad / "S0.nii.gz").astype(np.float32)
    D_w = _wls_D(trad)
    sl = _mid_slice(brain & valid)
    slice_mask = (brain & valid)[:, :, sl]
    xs, ys = np.where(slice_mask)
    if xs.size == 0:
        return
    X, Y, Z = brain.shape
    xs_lin = np.linspace(-1.0, 1.0, X, dtype=np.float32)
    ys_lin = np.linspace(-1.0, 1.0, Y, dtype=np.float32)
    zs_lin = np.linspace(-1.0, 1.0, Z, dtype=np.float32)
    xyz = np.stack([xs_lin[xs], ys_lin[ys], np.full(xs.shape, zs_lin[sl], np.float32)], axis=-1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ccfg = ckpt.get("config", {})
    model = SpatialDTIINR(
        hidden=int(ccfg.get("hidden", 128)),
        layers=int(ccfg.get("layers", 4)),
        pe_freqs=int(ccfg.get("pe_freqs", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    xyz_t = torch.from_numpy(xyz).to(device)
    S0_i, D_i = model(xyz_t)
    b0 = int(np.flatnonzero(bvals_u < float(cfg["b0_threshold"]))[0])
    dw_all = np.flatnonzero((bvals_u >= 800.0) & (bvals_u <= 1200.0))
    dw = int(dw_all[len(dw_all) // 2])
    bvals_2 = torch.from_numpy(bvals_u[[b0, dw]]).to(device)
    bvecs_2 = torch.from_numpy(bvecs_u[[b0, dw]]).to(device)
    pred_i = dti_forward_signal(S0_i, D_i, bvals_2, bvecs_2).cpu().numpy()
    S0_ws = torch.from_numpy(np.ascontiguousarray(S0_w[xs, ys, sl])).to(device)
    D_ws = torch.from_numpy(np.ascontiguousarray(D_w[xs, ys, sl])).to(device)
    pred_w = dti_forward_signal(S0_ws, D_ws, bvals_2, bvecs_2).cpu().numpy()
    obs = dwi[xs, ys, sl][:, [b0, dw]]

    def fill(vals: np.ndarray) -> np.ndarray:
        img = np.zeros((X, Y), dtype=np.float32)
        img[xs, ys] = vals.astype(np.float32)
        return img

    k = 1  # b≈1000
    vmax = float(np.percentile(obs[:, k], 99)) if obs.size else 1.0
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.8))
    _show(axes[0, 0], fill(obs[:, k]), "observed b=1000", 0, vmax)
    _show(axes[0, 1], fill(pred_i[:, k]), "INR pred", 0, vmax)
    _show(axes[0, 2], fill(np.abs(obs[:, k] - pred_i[:, k])), "|obs−INR|", 0, vmax * 0.4, "magma")
    _show(axes[1, 0], fill(obs[:, k]), "observed b=1000", 0, vmax)
    _show(axes[1, 1], fill(pred_w[:, k]), "WLS pred", 0, vmax)
    _show(axes[1, 2], fill(np.abs(obs[:, k] - pred_w[:, k])), "|obs−WLS|", 0, vmax * 0.4, "magma")
    fig.suptitle(f"{sid}  DWI slice z={sl}  vol={dw} (b≈1000)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dwi, dpi=150)
    plt.close(fig)

    npz = np.load(maps_npz)
    m = valid
    ad_w, rd_w = _nii(trad / "AD.nii.gz")[m], _nii(trad / "RD.nii.gz")[m]
    ad_i, rd_i = npz["AD"][m], npz["RD"][m]
    rng = np.random.default_rng(0)
    n = min(25000, int(ad_w.size))
    sel = rng.choice(ad_w.size, size=n, replace=False)
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    pairs = [
        (ad_w[sel] * 1e3, ad_i[sel] * 1e3, r"AD $\lambda_1$ ($\times 10^{-3}$)"),
        (rd_w[sel] * 1e3, rd_i[sel] * 1e3, r"RD $(\lambda_2+\lambda_3)/2$ ($\times 10^{-3}$)"),
        (ad_w[sel] * 1e3 - rd_w[sel] * 1e3, ad_i[sel] * 1e3 - rd_i[sel] * 1e3, r"anisotropy $\lambda_1-\mathrm{RD}$"),
    ]
    for ax, (xw, yi, title) in zip(axes, pairs):
        lo = float(min(np.min(xw), np.min(yi)))
        hi = float(max(np.max(xw), np.max(yi)))
        ax.plot([lo, hi], [lo, hi], color="#8a8680", lw=0.8, ls=":")
        ax.scatter(xw, yi, s=4, c="#2f5d50", alpha=0.25, rasterized=True)
        ax.set_xlabel("WLS")
        ax.set_ylabel("INR")
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
    fig.suptitle(f"{sid}  eigenvalues  (WLS vs INR)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_eig, dpi=150)
    plt.close(fig)
    del model, bundle, dwi
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description="WLS vs INR DWI RelMSE diagnostics")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--maps-only", action="store_true", help="Reuse diagnostic_table.csv; only make figures")
    args = ap.parse_args()

    cfg = load_config()
    exp = experiment_dir(cfg, "v1_schema_train_independent")
    trad_root = project_root() / "outputs" / "step1_traditional_dti"
    v0 = experiment_dir(cfg, "v0_preschema_independent")
    out = exp / "diagnostics"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    summary = _load_summary(exp / "summary.csv")
    if int(args.max_subjects) > 0:
        summary = summary[: int(args.max_subjects)]

    table_path = out / "diagnostic_table.csv"
    counts: dict[str, int] = {}
    if args.maps_only:
        if not table_path.is_file():
            raise SystemExit(f"--maps-only needs {table_path}")
        rows = _load_summary(table_path)
        for r in rows:
            counts[r.get("case", "")] = counts.get(r.get("case", ""), 0) + 1
        print(f"[diag] reused {table_path}  n={len(rows)}")
    else:
        rows = []
        for i, rec in enumerate(summary, 1):
            sid = rec["subject_id"]
            print(f"[{i}/{len(summary)}] WLS forward {sid}")
            trad = trad_root / sid
            bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
            bvals = bundle["bvals"]
            bvecs = normalize_bvecs(bvals, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
            vol_m = shell_volume_mask(
                bvals,
                b0_threshold=float(cfg["b0_threshold"]),
                shell_tol=float(cfg["shell_tol"]),
                shells=tuple(cfg.get("dti_shells", [1000.0])),
                include_b0=True,
            )
            dwi = np.ascontiguousarray(bundle["data"][..., vol_m], dtype=np.float32)
            brain = np.ascontiguousarray(bundle["brain_mask"])
            valid = np.ascontiguousarray(_nii(trad / "valid_mask.nii.gz") > 0.5)
            S0 = np.ascontiguousarray(_nii(trad / "S0.nii.gz"), dtype=np.float32)
            D = np.ascontiguousarray(_wls_D(trad))
            wls_raw = wls_dwi_relmse(
                S0=S0,
                D=D,
                dwi=dwi,
                mask=brain & valid,
                bvals=bvals[vol_m],
                bvecs=bvecs[vol_m],
                device=device,
            )
            sane = brain & valid & (S0 > S0_SANE_MIN) & (S0 < S0_SANE_MAX)
            wls_valid = wls_dwi_relmse(
                S0=S0,
                D=D,
                dwi=dwi,
                mask=sane,
                bvals=bvals[vol_m],
                bvecs=bvecs[vol_m],
                device=device,
            )
            fa = float(rec["FA_MAE"])
            inr = float(rec["DWI_RelMSE"])
            wls_r = float(wls_valid["relative_mse"])
            fa_s = _status_fa(fa)
            dwi_s = _status_dwi(inr)
            row = {
                "subject_id": sid,
                "WLS_DWI_RelMSE": wls_r,
                "WLS_DWI_MAE": float(wls_valid["MAE"]),
                "WLS_DWI_RelMSE_raw": float(wls_raw["relative_mse"]),
                "INR_DWI_RelMSE": inr,
                "INR_DWI_MAE": float(rec["DWI_MAE"]),
                "RelMSE_ratio_INR_over_WLS": (inr / wls_r) if wls_r > 0 else float("nan"),
                "FA_MAE": fa,
                "FA_r": float(rec["FA_r"]),
                "MD_MAE": float(rec["MD_MAE"]),
                "MD_r": float(rec["MD_r"]),
                "AD_MAE": float(rec["AD_MAE"]),
                "AD_r": float(rec["AD_r"]),
                "RD_MAE": float(rec["RD_MAE"]),
                "RD_r": float(rec["RD_r"]),
                "best_loss": float(rec["best_loss"]),
                "best_epoch": rec.get("best_epoch", ""),
                "n_voxels": rec.get("n_voxels", ""),
                "DWI_status": dwi_s,
                "FA_status": fa_s,
                "overall_status": _overall(fa_s, dwi_s),
                "case": _case(wls_r, inr, fa),
            }
            rows.append(row)
            print(f"    WLS RelMSE={wls_r:.4f}  INR RelMSE={inr:.4f}  FA_MAE={fa:.3f}  {row['case']}")
            del bundle, dwi
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fields = list(rows[0].keys())
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        save_json(out / "diagnostic_table.json", rows)
        for r in rows:
            counts[r["case"]] = counts.get(r["case"], 0) + 1
        lines = [
            "# Independent INR diagnostics (WLS vs INR DWI)",
            "",
            f"- N = {len(rows)}",
            f"- FA failure threshold (experimental): MAE ≥ {FA_FAIL}",
            f"- INR DWI failure threshold (experimental): RelMSE ≥ {DWI_FAIL}",
            f"- WLS signal-bad threshold (experimental): RelMSE ≥ {WLS_BAD}",
            f"- WLS DWI RelMSE uses brain ∩ valid ∩ ({S0_SANE_MIN} < S0_hat < {S0_SANE_MAX:.0f}); raw column is unfiltered and can be dominated by <1% exploding S0_hat voxels.",
            "- INR RelMSE uses brain voxels (same seed=42, cap=131072).",
            "",
            "| subject | WLS DWI RelMSE | INR DWI RelMSE | FA MAE | FA r | MD MAE | AD MAE | RD MAE | best_loss | overall | case |",
            "|---------|---------------:|---------------:|-------:|-----:|-------:|-------:|-------:|----------:|---------|------|",
        ]
        for r in sorted(rows, key=lambda z: -float(z["FA_MAE"])):
            mark = " **" if r["subject_id"] in FAILURES else ""
            lines.append(
                f"| {r['subject_id']}{mark} | {float(r['WLS_DWI_RelMSE']):.4f} | {float(r['INR_DWI_RelMSE']):.4f} | "
                f"{float(r['FA_MAE']):.3f} | {float(r['FA_r']):.3f} | {float(r['MD_MAE']):.6f} | "
                f"{float(r['AD_MAE']):.6f} | {float(r['RD_MAE']):.6f} | {float(r['best_loss']):.4f} | "
                f"{r['overall_status']} | {r['case']} |"
            )
        lines += ["", "## Case counts", ""]
        for k, v in sorted(counts.items()):
            lines.append(f"- `{k}`: {v}")
        lines.append("")
        (out / "diagnostic_table.md").write_text("\n".join(lines), encoding="utf-8")

    plot_wls_vs_inr(rows, fig_dir / "wls_vs_inr_dwi.png")
    plot_training_curves(v0, exp, fig_dir / "training_curves_failures.png")

    if not args.skip_maps:
        for sid in (*FAILURES, CONTROL):
            print(f"[maps] {sid}")
            trad = trad_root / sid
            maps = exp / sid / "maps.npz"
            ckpt = exp / sid / "best.pt"
            plot_param_maps(sid, trad, maps, fig_dir / f"maps_{sid}.png")
            plot_dwi_and_evals(
                sid=sid,
                cfg=cfg,
                trad=trad,
                ckpt_path=ckpt,
                maps_npz=maps,
                device=device,
                out_dwi=fig_dir / f"dwi_{sid}.png",
                out_eig=fig_dir / f"eigen_{sid}.png",
            )

    print(f"\n[diag] wrote {table_path}")
    print("case counts:", counts)


if __name__ == "__main__":
    main()
