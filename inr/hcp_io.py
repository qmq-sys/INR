"""Load HCP diffusion subjects (ConnectomeDB-style layout)."""
from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np

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


def _find_dwi_nifti(diffusion_dir: str) -> str:
    for name in ("data.nii.gz", "data_1.25mm.nii.gz"):
        p = os.path.join(diffusion_dir, name)
        if os.path.isfile(p):
            return p
    cands = [
        p
        for p in glob.glob(os.path.join(diffusion_dir, "*.nii.gz"))
        if "mask" not in os.path.basename(p).lower()
    ]
    if not cands:
        raise FileNotFoundError(f"No DWI NIfTI in {diffusion_dir}")
    cands.sort(key=lambda p: len(os.path.basename(p)))
    return cands[0]


def _read_bvals_bvecs(path_bvals: str, path_bvecs: str) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.asarray(np.loadtxt(path_bvals), dtype=np.float64).ravel()
    bvecs = np.asarray(np.loadtxt(path_bvecs), dtype=np.float64)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError(f"bvecs shape {bvecs.shape}, expected [N,3]")
    if bvals.shape[0] != bvecs.shape[0]:
        raise ValueError(f"bvals len {bvals.shape[0]} != bvecs N {bvecs.shape[0]}")
    return bvals.astype(np.float32), bvecs.astype(np.float32)


def resolve_diffusion_dir(hcp_root: str, subject_id: str, diffusion_subdir: str = "T1w/Diffusion") -> str:
    root = os.path.expanduser(hcp_root)
    sub = str(subject_id).strip()
    candidates = [
        os.path.join(root, sub, diffusion_subdir),
        os.path.join(root, sub, sub, diffusion_subdir),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    hits = [
        p
        for p in glob.glob(os.path.join(root, sub, "**", "T1w", "Diffusion"), recursive=True)
        if os.path.isdir(p)
    ]
    if hits:
        hits.sort(key=len)
        return hits[0]
    raise FileNotFoundError(f"Diffusion dir not found for subject {sub} under {root}")


def inventory_hcp_subject(
    hcp_root: str,
    subject_id: str,
    *,
    b0_threshold: float = 50.0,
) -> dict[str, Any]:
    """Lightweight metadata (no full DWI load)."""
    _require_nib()
    diff_dir = resolve_diffusion_dir(hcp_root, subject_id)
    dwi_path = _find_dwi_nifti(diff_dir)
    bvals_path = os.path.join(diff_dir, "bvals")
    bvecs_path = os.path.join(diff_dir, "bvecs")
    mask_path = os.path.join(diff_dir, "nodif_brain_mask.nii.gz")

    img = nib.load(dwi_path)
    shape = tuple(int(x) for x in img.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected 4D DWI, got shape={shape}")
    bvals, bvecs = _read_bvals_bvecs(bvals_path, bvecs_path)
    if bvals.shape[0] != shape[3]:
        raise ValueError(f"N volumes {shape[3]} != bvals {bvals.shape[0]}")
    b0_mask = bvals < float(b0_threshold)
    if not np.any(b0_mask):
        raise RuntimeError("No b0 volumes found")

    n_brain = 0
    if os.path.isfile(mask_path):
        brain_mask = np.asanyarray(nib.load(mask_path).dataobj) > 0.5
        if brain_mask.shape != shape[:3]:
            raise ValueError("mask / data spatial mismatch")
        n_brain = int(brain_mask.sum())

    return {
        "subject_id": str(subject_id).strip(),
        "bvals": bvals,
        "bvecs": bvecs,
        "b0_mask": b0_mask,
        "shape": shape,
        "n_brain_voxels": n_brain,
        "diffusion_dir": diff_dir,
        "dwi_path": dwi_path,
        "bvals_path": bvals_path,
        "bvecs_path": bvecs_path,
        "mask_path": mask_path if os.path.isfile(mask_path) else "",
    }


def load_hcp_subject(
    hcp_root: str,
    subject_id: str,
    *,
    b0_threshold: float = 50.0,
) -> dict[str, Any]:
    """
    Returns:
      data [X,Y,Z,N], bvals [N], bvecs [N,3], brain_mask [X,Y,Z],
      S0 [X,Y,Z], affine [4,4], paths...
    """
    _require_nib()
    meta = inventory_hcp_subject(hcp_root, subject_id, b0_threshold=b0_threshold)
    img = nib.load(meta["dwi_path"])
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    bvals = meta["bvals"]
    b0_mask = meta["b0_mask"]

    mask_path = meta["mask_path"]
    if mask_path and os.path.isfile(mask_path):
        brain_mask = np.asanyarray(nib.load(mask_path).dataobj) > 0.5
    else:
        s0_tmp = data[..., b0_mask].mean(axis=-1)
        brain_mask = s0_tmp > (0.1 * float(np.max(s0_tmp)))

    S0 = data[..., b0_mask].mean(axis=-1).astype(np.float32)

    return {
        "subject_id": meta["subject_id"],
        "data": data,
        "bvals": bvals,
        "bvecs": meta["bvecs"],
        "brain_mask": brain_mask.astype(bool),
        "S0": S0,
        "b0_mask": b0_mask,
        "affine": affine,
        "diffusion_dir": meta["diffusion_dir"],
        "dwi_path": meta["dwi_path"],
        "bvals_path": meta["bvals_path"],
        "bvecs_path": meta["bvecs_path"],
        "mask_path": mask_path,
    }


def normalize_bvecs(bvals: np.ndarray, bvecs: np.ndarray, *, b0_threshold: float = 50.0) -> np.ndarray:
    bvals = np.asarray(bvals, dtype=np.float64).ravel()
    bvecs = np.asarray(bvecs, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    b0 = bvals < float(b0_threshold)
    norms = np.where((~b0.reshape(-1, 1)) & (norms > 0), norms, 1.0)
    out = bvecs / norms
    out[b0] = 0.0
    return out.astype(np.float32)


def shell_volume_mask(
    bvals: np.ndarray,
    *,
    b0_threshold: float = 50.0,
    shell_tol: float = 200.0,
    shells: tuple[float, ...] | list[float] = (1000.0,),
    include_b0: bool = True,
) -> np.ndarray:
    b = np.asarray(bvals, dtype=np.float64).ravel()
    b0 = b < float(b0_threshold)
    m = b0.copy() if include_b0 else np.zeros_like(b0)
    for tb in shells:
        m |= (~b0) & (np.abs(b - float(tb)) <= float(shell_tol))
    return m
