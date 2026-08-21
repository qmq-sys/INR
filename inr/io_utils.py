"""Small I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    nib = None
    _NIB_ERR = e
else:
    _NIB_ERR = None


def _require_nib() -> None:
    if nib is None:
        raise ImportError(f"nibabel required: {_NIB_ERR}")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else project_root() / "config" / "default.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def experiment_dir(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a named experiment output directory from config.experiments."""
    mapping = cfg.get("experiments") or {}
    rel = mapping.get(key)
    if not rel:
        raise KeyError(f"config.experiments missing key: {key}")
    return project_root() / str(cfg.get("output_root", "outputs")) / str(rel)


def load_subjects_yaml(path: str | Path) -> list[str]:
    """Load subject IDs from config/subjects.yaml (key: subjects)."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "subjects" not in obj:
        raise ValueError(f"Expected key 'subjects' in {p}")
    return [str(s).strip() for s in obj["subjects"] if str(s).strip()]


def resolve_subject_list(
    *,
    subjects_csv: str = "",
    subjects_yaml: str | Path | None = None,
    subjects_file: str | Path | None = None,
) -> list[str]:
    if subjects_csv.strip():
        return [s.strip() for s in subjects_csv.split(",") if s.strip()]
    if subjects_yaml is not None:
        p = Path(subjects_yaml)
        if p.is_file():
            return load_subjects_yaml(p)
    if subjects_file is not None:
        return read_subjects(subjects_file)
    raise FileNotFoundError("No subjects source provided")


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_nifti(path: str | Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    _require_nib()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4, dtype=np.float32) if affine is None else np.asarray(affine, dtype=np.float32)
    nib.save(nib.Nifti1Image(np.asarray(data), aff), str(p))


def count_shells(bvals: np.ndarray, *, b0_threshold: float = 50.0, shell_tol: float = 200.0) -> dict[str, Any]:
    b = np.asarray(bvals, dtype=np.float64).ravel()
    b0 = b < float(b0_threshold)
    shells = {
        "n_b0": int(np.count_nonzero(b0)),
        "n_b1000": int(np.count_nonzero((~b0) & (np.abs(b - 1000.0) <= shell_tol))),
        "n_b2000": int(np.count_nonzero((~b0) & (np.abs(b - 2000.0) <= shell_tol))),
        "n_b3000": int(np.count_nonzero((~b0) & (np.abs(b - 3000.0) <= shell_tol))),
        "n_total": int(b.size),
        "unique_bvals": sorted({float(x) for x in np.round(b, 1).tolist()}),
    }
    non_b0 = b[~b0]
    shells["n_diffusion_dirs"] = int(non_b0.size)
    shells["is_multi_shell"] = bool(shells["n_b1000"] > 0 and shells["n_b2000"] > 0)
    shells["shell_type"] = "multi-shell" if shells["is_multi_shell"] else ("single-shell" if shells["n_b1000"] > 0 else "unknown")
    return shells
