"""Fixed evaluation schema for all INR experiments (Independent / Shared / Latent / ...).

Three metric families only:
  1) Parameter agreement vs WLS reference: FA/MD/AD/RD  MAE, RMSE, Pearson r
  2) DWI reconstruction: MAE, RelMSE
  3) Training stability: final_loss, best_loss, best_epoch, training_time_sec

Per-subject default files:
  best.pt / maps.npz / metrics.json

Experiment-level:
  summary.csv / aggregate.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import save_json

PARAM_KEYS = ("FA", "MD", "AD", "RD")

# Flat columns for summary.csv (Experiment Master Table)
SUMMARY_COLUMNS = [
    "subject_id",
    "ok",
    "error",
    "FA_MAE",
    "FA_RMSE",
    "FA_r",
    "MD_MAE",
    "MD_RMSE",
    "MD_r",
    "AD_MAE",
    "AD_RMSE",
    "AD_r",
    "RD_MAE",
    "RD_RMSE",
    "RD_r",
    "DWI_MAE",
    "DWI_RelMSE",
    "final_loss",
    "best_loss",
    "best_epoch",
    "training_time_sec",
    "n_voxels",
    "epochs",
]

AGGREGATE_METRICS = [
    "FA_MAE",
    "FA_RMSE",
    "FA_r",
    "MD_MAE",
    "MD_RMSE",
    "MD_r",
    "AD_MAE",
    "AD_RMSE",
    "AD_r",
    "RD_MAE",
    "RD_RMSE",
    "RD_r",
    "DWI_MAE",
    "DWI_RelMSE",
    "final_loss",
    "best_loss",
    "training_time_sec",
]


def _masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)[np.asarray(mask, dtype=bool)]


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = _masked(a, mask), _masked(b, mask)
    return float("nan") if x.size == 0 else float(np.mean(np.abs(x - y)))


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = _masked(a, mask), _masked(b, mask)
    return float("nan") if x.size == 0 else float(np.sqrt(np.mean((x - y) ** 2)))


def pearson_r(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = _masked(a, mask), _masked(b, mask)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def parameter_agreement_vs_wls(
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Agreement with conventional WLS reference (NOT ground-truth error)."""
    m = np.asarray(mask, dtype=bool)
    out: dict[str, Any] = {"n_voxels": int(np.count_nonzero(m)), "reference": "WLS-DTI"}
    for key in PARAM_KEYS:
        out[key] = {
            "MAE": mae(pred[key], ref[key], m),
            "RMSE": rmse(pred[key], ref[key], m),
            "Pearson": pearson_r(pred[key], ref[key], m),
        }
    return out


def dwi_reconstruction_metrics(
    pred: np.ndarray | Any,
    obs: np.ndarray | Any,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    RelMSE = ||pred-obs||_2^2 / (||obs||_2^2 + eps)
    MAE    = mean |pred-obs|
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    diff = p - o
    return {
        "MAE": float(np.mean(np.abs(diff))),
        "relative_mse": float(np.sum(diff * diff) / (np.sum(o * o) + float(eps))),
        "n_values": int(p.size),
    }


def build_metrics_json(
    *,
    subject_id: str,
    parameter_metrics: dict[str, Any],
    dwi: dict[str, float],
    training: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj = {
        "subject_id": str(subject_id),
        "schema_version": 1,
        "note": (
            "parameter_metrics are agreement with conventional WLS-DTI reference "
            "(not ground-truth error)."
        ),
        "parameter_metrics": {
            k: parameter_metrics[k] for k in PARAM_KEYS if k in parameter_metrics
        },
        "dwi": {
            "MAE": float(dwi["MAE"]),
            "relative_mse": float(dwi["relative_mse"]),
        },
        "training": {
            "final_loss": training.get("final_loss"),
            "best_loss": training.get("best_loss"),
            "best_epoch": training.get("best_epoch"),
            "training_time_sec": training.get("training_time_sec"),
            "epochs": training.get("epochs"),
        },
        "n_voxels": parameter_metrics.get("n_voxels"),
    }
    if extra:
        obj["extra"] = extra
    return obj


def metrics_json_to_summary_row(metrics: dict[str, Any], *, ok: bool = True, error: str = "") -> dict[str, Any]:
    pm = metrics.get("parameter_metrics", {})
    dwi = metrics.get("dwi", {})
    tr = metrics.get("training", {})
    row: dict[str, Any] = {
        "subject_id": metrics.get("subject_id"),
        "ok": ok,
        "error": error,
        "DWI_MAE": dwi.get("MAE"),
        "DWI_RelMSE": dwi.get("relative_mse"),
        "final_loss": tr.get("final_loss"),
        "best_loss": tr.get("best_loss"),
        "best_epoch": tr.get("best_epoch"),
        "training_time_sec": tr.get("training_time_sec"),
        "n_voxels": metrics.get("n_voxels"),
        "epochs": tr.get("epochs"),
    }
    for k in PARAM_KEYS:
        block = pm.get(k, {})
        row[f"{k}_MAE"] = block.get("MAE")
        row[f"{k}_RMSE"] = block.get("RMSE")
        row[f"{k}_r"] = block.get("Pearson")
    return row


def _stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def write_summary_and_aggregate(out_root: Path, rows: list[dict[str, Any]]) -> None:
    """Write summary.csv + aggregate.csv (+ short markdown)."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUMMARY_COLUMNS})

    ok_rows = [r for r in rows if str(r.get("ok", True)).lower() in {"1", "true", "yes", "True"}]
    agg_rows = []
    for key in AGGREGATE_METRICS:
        st = _stats([r.get(key) for r in ok_rows])
        agg_rows.append({"metric": key, **st, "mean_pm_std": f"{st['mean']:.6g} ± {st['std']:.6g}"})

    with open(out_root / "aggregate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "median", "min", "max", "n", "mean_pm_std"])
        w.writeheader()
        w.writerows(agg_rows)

    save_json(out_root / "aggregate.json", {"n_subjects": len(ok_rows), "metrics": {r["metric"]: r for r in agg_rows}})

    lines = [
        "# Experiment aggregate (mean ± std)",
        "",
        f"- N subjects (ok): **{len(ok_rows)}**",
        "- Parameter MAE/RMSE/r = agreement vs WLS reference (not GT error)",
        "",
        "| Metric | mean ± std | median |",
        "|--------|-----------:|-------:|",
    ]
    for r in agg_rows:
        lines.append(f"| {r['metric']} | {r['mean_pm_std']} | {r['median']:.6g} |")
    lines.append("")
    (out_root / "aggregate.md").write_text("\n".join(lines), encoding="utf-8")


def collect_metrics_from_experiment(exp_root: Path) -> list[dict[str, Any]]:
    """Load all <sid>/metrics.json under an experiment root into summary rows."""
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(exp_root).glob("*/metrics.json")):
        import json

        obj = json.loads(p.read_text(encoding="utf-8"))
        rows.append(metrics_json_to_summary_row(obj, ok=True))
    return rows
