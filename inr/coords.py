"""Coordinate helpers for spatial INR."""
from __future__ import annotations

import numpy as np
import torch


def voxel_coords_normalized(shape_xyz: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    """All voxel centers as [X*Y*Z, 3] in [-1,1]^3."""
    X, Y, Z = [int(v) for v in shape_xyz]
    xs = np.linspace(-1.0, 1.0, X, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, Y, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, Z, dtype=np.float32)
    # meshgrid indexing='ij' matches array axis order
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def masked_coords_and_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      coords_m11: [V,3] in [-1,1]
      flat_idx:   [V] flat indices into X*Y*Z
    """
    mask = np.asarray(mask, dtype=bool)
    X, Y, Z = mask.shape
    coords_all = voxel_coords_normalized((X, Y, Z))
    flat = np.flatnonzero(mask.reshape(-1))
    return coords_all[flat], flat


def sample_batch_indices(n: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    batch = int(min(max(1, batch), n))
    return rng.integers(0, n, size=batch, endpoint=False)
