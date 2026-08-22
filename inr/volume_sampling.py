"""Deterministic DWI volume subsampling for Experiment 3 (low sampling).

Reduces acquisition volumes (b0 + b≈1000), NOT spatial voxels.
Nested subsets: 10% ⊂ 25% ⊂ 50% ⊂ 100% from one seeded shuffle per category.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .hcp_io import load_hcp_subject, shell_volume_mask
from .io_utils import save_json

DEFAULT_SEED = 42
DEFAULT_LEVELS: tuple[tuple[str, float], ...] = (
    ("100%", 1.0),
    ("50%", 0.5),
    ("25%", 0.25),
    ("10%", 0.1),
)


def build_volume_sampling_protocol(
    bvals: np.ndarray,
    vol_mask: np.ndarray,
    *,
    b0_threshold: float = 50.0,
    seed: int = DEFAULT_SEED,
    levels: tuple[tuple[str, float], ...] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """
    Build nested volume index sets relative to ``vol_mask``-selected volumes.

    Returns protocol dict with keys like ``100%``, ``50%``, each containing:
      indices, n_b0, n_b1000, n_total, fraction
    """
    bvals_u = np.asarray(bvals, dtype=np.float64).ravel()[np.asarray(vol_mask, dtype=bool)]
    n = int(bvals_u.size)
    b0_idx = np.flatnonzero(bvals_u < float(b0_threshold)).astype(np.int64)
    dw_idx = np.flatnonzero(bvals_u >= float(b0_threshold)).astype(np.int64)
    if b0_idx.size == 0 or dw_idx.size == 0:
        raise ValueError(f"need both b0 and dw volumes; got b0={b0_idx.size} dw={dw_idx.size}")

    rng = np.random.default_rng(int(seed))
    b0_ord = b0_idx[rng.permutation(b0_idx.size)]
    dw_ord = dw_idx[rng.permutation(dw_idx.size)]

    out_levels: dict[str, Any] = {}
    for label, frac in levels:
        if float(frac) >= 1.0 - 1e-9:
            pick = np.sort(np.concatenate([b0_idx, dw_idx]))
        else:
            n_b0 = max(1, int(round(b0_idx.size * float(frac))))
            n_dw = max(1, int(round(dw_idx.size * float(frac))))
            pick = np.sort(np.concatenate([b0_ord[:n_b0], dw_ord[:n_dw]]))
        out_levels[label] = {
            "fraction": float(frac),
            "indices": [int(x) for x in pick.tolist()],
            "n_total": int(pick.size),
            "n_b0": int(np.count_nonzero(bvals_u[pick] < float(b0_threshold))),
            "n_b1000": int(np.count_nonzero(bvals_u[pick] >= float(b0_threshold))),
            "n_b0_full": int(b0_idx.size),
            "n_b1000_full": int(dw_idx.size),
            "n_total_full": int(n),
        }

    return {
        "seed": int(seed),
        "b0_threshold": float(b0_threshold),
        "description": "Nested volume subsets within shell_volume_mask(b0+b1000); indices relative to masked volume axis.",
        "levels": out_levels,
    }


def protocol_from_config_reference(
    cfg: dict[str, Any],
    *,
    reference_subject: str = "101309",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    bundle = load_hcp_subject(cfg["hcp_root"], reference_subject, b0_threshold=float(cfg["b0_threshold"]))
    vol_m = shell_volume_mask(
        bundle["bvals"],
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    proto = build_volume_sampling_protocol(
        bundle["bvals"],
        vol_m,
        b0_threshold=float(cfg["b0_threshold"]),
        seed=seed,
    )
    proto["reference_subject"] = str(reference_subject)
    proto["shell_tol"] = float(cfg["shell_tol"])
    proto["dti_shells"] = list(cfg.get("dti_shells", [1000.0]))
    return proto


def save_sampling_protocol(path: Path, protocol: dict[str, Any]) -> None:
    save_json(path, protocol)


def load_sampling_protocol(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def level_indices(protocol: dict[str, Any], level_label: str) -> np.ndarray:
    levels = protocol["levels"]
    if level_label not in levels:
        raise KeyError(f"unknown level {level_label!r}; have {list(levels)}")
    return np.asarray(levels[level_label]["indices"], dtype=np.int64)


def level_fraction(protocol: dict[str, Any], level_label: str) -> float:
    return float(protocol["levels"][level_label]["fraction"])


def pct_dir_name(level_label: str) -> str:
    """Map '50%' → 'pct50'."""
    return "pct" + level_label.replace("%", "")


def label_from_pct_dir(name: str) -> str:
    if name.startswith("pct") and name.endswith("%"):
        return name
    if name.startswith("pct"):
        return name[3:] + "%"
    return name
