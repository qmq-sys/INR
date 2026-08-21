"""Shared single-subject Independent-INR training (Experiment 1 & 2).

Physics:
  (x,y,z) -> SpatialDTIINR -> (S0, D) -> DTI forward -> relative MSE(DWI)

Default outputs per subject (fixed schema):
  best.pt
  maps.npz          # S0, FA, MD, AD, RD
  metrics.json      # FA/MD/AD/RD MAE/RMSE/r + DWI MAE/RelMSE + training

Optional:
  --save_nifti  -> nifti/FA.nii.gz ...
  --save_tensor -> tensor.npz (D + lower-triangular comps)
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .coords import masked_coords_and_indices
from .dti_fit import fit_dti_b0_b1000, tensor_to_lower6
from .hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask
from .io_utils import save_json, save_nifti
from .metrics_schema import (
    build_metrics_json,
    dwi_reconstruction_metrics,
    metrics_json_to_summary_row,
    parameter_agreement_vs_wls,
)
from .model import SpatialDTIINR
from .physics import compute_fa_md_ad_rd, dti_forward_signal


def resolve_device(name: str) -> torch.device:
    if name in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_or_fit_wls_reference(
    *,
    bundle: dict[str, Any],
    trad_dir: Path,
    cfg: dict[str, Any],
    skip_if_exists: bool,
) -> dict[str, Any]:
    trad_dir = Path(trad_dir)
    ref_fa = trad_dir / "FA.nii.gz"
    affine = bundle["affine"]
    brain = bundle["brain_mask"]

    if skip_if_exists and ref_fa.is_file():
        import nibabel as nib

        return {
            "FA": np.asanyarray(nib.load(trad_dir / "FA.nii.gz").dataobj, dtype=np.float32),
            "MD": np.asanyarray(nib.load(trad_dir / "MD.nii.gz").dataobj, dtype=np.float32),
            "AD": np.asanyarray(nib.load(trad_dir / "AD.nii.gz").dataobj, dtype=np.float32),
            "RD": np.asanyarray(nib.load(trad_dir / "RD.nii.gz").dataobj, dtype=np.float32),
            "valid_mask": np.asanyarray(nib.load(trad_dir / "valid_mask.nii.gz").dataobj) > 0.5,
            "D": np.asanyarray(nib.load(trad_dir / "D.nii.gz").dataobj, dtype=np.float32),
            "S0": np.asanyarray(nib.load(trad_dir / "S0.nii.gz").dataobj, dtype=np.float32),
        }

    ref = fit_dti_b0_b1000(
        bundle["data"],
        bundle["bvals"],
        bundle["bvecs"],
        brain,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
    )
    trad_dir.mkdir(parents=True, exist_ok=True)
    # Traditional baseline keeps NIfTI (used as shared reference across experiments)
    for k in ("S0", "D", "FA", "MD", "AD", "RD"):
        save_nifti(trad_dir / f"{k}.nii.gz", ref[k], affine)
    comps = tensor_to_lower6(ref["D"])
    for k, v in comps.items():
        save_nifti(trad_dir / f"{k}.nii.gz", v, affine)
    save_nifti(trad_dir / "valid_mask.nii.gz", ref["valid_mask"].astype("uint8"), affine)
    save_nifti(trad_dir / "brain_mask.nii.gz", brain.astype("uint8"), affine)
    return ref


@torch.no_grad()
def predict_maps(
    model: SpatialDTIINR,
    coords: torch.Tensor,
    flat_idx: np.ndarray,
    shape_xyz: tuple[int, int, int],
    device: torch.device,
    *,
    chunk: int = 65536,
    want_D: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    X, Y, Z = shape_xyz
    n = coords.shape[0]
    S0_vol = np.zeros((X * Y * Z,), dtype=np.float32)
    FA = np.zeros((X * Y * Z,), dtype=np.float32)
    MD = np.zeros((X * Y * Z,), dtype=np.float32)
    AD = np.zeros((X * Y * Z,), dtype=np.float32)
    RD = np.zeros((X * Y * Z,), dtype=np.float32)
    D_vol = np.zeros((X * Y * Z, 3, 3), dtype=np.float32) if want_D else None

    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        xyz = coords[sl].to(device)
        S0, D = model(xyz)
        fa, md, ad, rd = compute_fa_md_ad_rd(D.detach().float().cpu())
        idx = flat_idx[sl]
        S0_vol[idx] = S0.detach().float().cpu().numpy()
        FA[idx] = fa.numpy()
        MD[idx] = md.numpy()
        AD[idx] = ad.numpy()
        RD[idx] = rd.numpy()
        if D_vol is not None:
            D_vol[idx] = D.detach().float().cpu().numpy()

    out = {
        "S0": S0_vol.reshape(X, Y, Z),
        "FA": FA.reshape(X, Y, Z),
        "MD": MD.reshape(X, Y, Z),
        "AD": AD.reshape(X, Y, Z),
        "RD": RD.reshape(X, Y, Z),
    }
    if D_vol is not None:
        out["D"] = D_vol.reshape(X, Y, Z, 3, 3)
        out.update(tensor_to_lower6(out["D"]))
    return out


@torch.no_grad()
def evaluate_dwi_reconstruction(
    model: SpatialDTIINR,
    coords: torch.Tensor,
    flat_idx: np.ndarray,
    dwi_flat: np.ndarray,
    bvals_t: torch.Tensor,
    bvecs_t: torch.Tensor,
    device: torch.device,
    *,
    max_voxels: int = 131072,
    seed: int = 0,
    evaluation_mask: str = "brain & WLS_valid",
) -> dict[str, float]:
    """Global RelMSE = ||pred-obs||^2 / (||obs||^2 + eps); plus MAE.

    ``coords`` / ``flat_idx`` must already correspond to the evaluation mask
    (official protocol: brain & WLS_valid). RelMSE math is unchanged.
    """
    model.eval()
    n_eval = int(coords.shape[0])
    rng = np.random.default_rng(seed)
    sel = np.arange(n_eval) if n_eval <= max_voxels else rng.choice(n_eval, size=max_voxels, replace=False)
    n_sampled = int(sel.size)
    xyz = coords[sel].to(device)
    target = torch.from_numpy(dwi_flat[flat_idx[sel]]).to(device)
    S0, D = model(xyz)
    pred = dti_forward_signal(S0, D, bvals_t, bvecs_t)
    out = dwi_reconstruction_metrics(pred.detach().float().cpu().numpy(), target.detach().float().cpu().numpy())
    out["evaluation_mask"] = str(evaluation_mask)
    out["max_voxels"] = float(max_voxels)
    out["seed"] = float(seed)
    out["n_eval_voxels"] = float(n_eval)
    out["n_sampled_voxels"] = float(n_sampled)
    return out


def save_subject_outputs(
    *,
    out_dir: Path,
    sid: str,
    model: SpatialDTIINR,
    maps: dict[str, np.ndarray],
    metrics_obj: dict[str, Any],
    ckpt_payload: dict[str, Any],
    affine: np.ndarray,
    save_nifti_flag: bool,
    save_tensor_flag: bool,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ckpt_payload, out_dir / "best.pt")
    np.savez_compressed(
        out_dir / "maps.npz",
        S0=maps["S0"].astype(np.float32),
        FA=maps["FA"].astype(np.float32),
        MD=maps["MD"].astype(np.float32),
        AD=maps["AD"].astype(np.float32),
        RD=maps["RD"].astype(np.float32),
    )
    save_json(out_dir / "metrics.json", metrics_obj)

    if save_tensor_flag:
        if "D" not in maps:
            raise RuntimeError("save_tensor requested but D not in maps")
        np.savez_compressed(
            out_dir / "tensor.npz",
            D=maps["D"].astype(np.float32),
            **{k: maps[k].astype(np.float32) for k in ("Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz") if k in maps},
        )

    if save_nifti_flag:
        nifti_dir = out_dir / "nifti"
        for k in ("S0", "FA", "MD", "AD", "RD"):
            save_nifti(nifti_dir / f"{k}.nii.gz", maps[k], affine)


def train_one_independent_subject(
    *,
    subject_id: str,
    cfg: dict[str, Any],
    out_dir: Path,
    trad_dir: Path,
    device: torch.device,
    epochs: int,
    batch_voxels: int,
    lr: float,
    hidden: int,
    layers: int,
    pe_freqs: int,
    log_every: int,
    eval_every: int,
    seed: int,
    skip_traditional_if_exists: bool = True,
    save_nifti_flag: bool = False,
    save_tensor_flag: bool = False,
    tag: str = "IndependentINR",
) -> dict[str, Any]:
    """
    Train a *fresh* SpatialDTIINR for one subject.
    Returns a flat summary-row dict (for summary.csv).
    """
    sid = str(subject_id).strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[{tag}] subject={sid} device={device} epochs={epochs}")
    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=float(cfg["b0_threshold"]))
    data = bundle["data"]
    bvals = bundle["bvals"]
    bvecs = normalize_bvecs(bvals, bundle["bvecs"], b0_threshold=float(cfg["b0_threshold"]))
    brain = bundle["brain_mask"]
    affine = bundle["affine"]

    vol_m = shell_volume_mask(
        bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        shell_tol=float(cfg["shell_tol"]),
        shells=tuple(cfg.get("dti_shells", [1000.0])),
        include_b0=True,
    )
    dwi = data[..., vol_m].astype(np.float32)
    bvals_u = bvals[vol_m].astype(np.float32)
    bvecs_u = bvecs[vol_m].astype(np.float32)
    print(f"[{tag}] {sid}: using {int(vol_m.sum())} volumes (b0+b1000)")

    ref = load_or_fit_wls_reference(
        bundle=bundle,
        trad_dir=trad_dir,
        cfg=cfg,
        skip_if_exists=skip_traditional_if_exists,
    )

    # Training mask stays brain-only (unchanged). Evaluation uses common_mask below.
    train_coords_np, train_flat_idx = masked_coords_and_indices(brain)
    dwi_flat = dwi.reshape(-1, dwi.shape[-1])
    n_vox = int(train_coords_np.shape[0])
    print(f"[{tag}] {sid}: brain voxels={n_vox}")

    model = SpatialDTIINR(hidden=hidden, layers=layers, pe_freqs=pe_freqs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_coords_t = torch.from_numpy(train_coords_np)
    bvals_t = torch.from_numpy(bvals_u).to(device)
    bvecs_t = torch.from_numpy(bvecs_u).to(device)

    t0 = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    final_loss = float("nan")

    model.train()
    for epoch in range(1, epochs + 1):
        n_steps = max(1, int(np.ceil(n_vox / batch_voxels)))
        losses: list[float] = []
        for _ in range(n_steps):
            sel = rng.integers(0, n_vox, size=batch_voxels, endpoint=False)
            xyz = train_coords_t[sel].to(device)
            idx = train_flat_idx[sel]
            target = torch.from_numpy(dwi_flat[idx]).to(device)

            S0, D = model(xyz)
            pred = dti_forward_signal(S0, D, bvals_t, bvecs_t)
            s0_obs = target[:, bvals_t < float(cfg["b0_threshold"])].mean(dim=-1, keepdim=True).clamp_min(1.0)
            loss = torch.mean(((pred - target) / s0_obs) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        final_loss = mean_loss
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"  [{sid}] epoch {epoch:4d}/{epochs}  L_DWI={mean_loss:.6e}  best={best_loss:.6e}@{best_epoch}")

    # Restore best weights for evaluation / save
    if best_state is not None:
        model.load_state_dict(best_state)

    maps = predict_maps(
        model,
        train_coords_t,
        train_flat_idx,
        data.shape[:3],
        device,
        want_D=bool(save_tensor_flag),
    )
    # Official evaluation protocol: FA/MD/AD/RD and DWI all use common_mask.
    common_mask = brain & ref["valid_mask"]
    if not np.all(~common_mask | brain):
        raise RuntimeError(f"{sid}: common_mask is not a subset of brain")
    n_brain = int(np.count_nonzero(brain))
    n_wls_valid = int(np.count_nonzero(ref["valid_mask"]))
    n_common = int(np.count_nonzero(common_mask))
    print(f"[EvalMask] brain: {n_brain}  WLS_valid: {n_wls_valid}  common: {n_common}")
    if n_common == 0:
        raise RuntimeError(f"{sid}: empty common_mask (brain & WLS_valid)")
    if n_common > n_brain:
        raise RuntimeError(f"{sid}: common_mask larger than brain ({n_common} > {n_brain})")

    eval_coords_np, eval_flat_idx = masked_coords_and_indices(common_mask)
    eval_coords_t = torch.from_numpy(eval_coords_np)
    param_metrics = parameter_agreement_vs_wls(maps, ref, common_mask)
    dwi_metrics = evaluate_dwi_reconstruction(
        model,
        eval_coords_t,
        eval_flat_idx,
        dwi_flat,
        bvals_t,
        bvecs_t,
        device,
        seed=seed,
        evaluation_mask="brain & WLS_valid",
    )
    train_time = time.time() - t0
    metrics_obj = build_metrics_json(
        subject_id=sid,
        parameter_metrics=param_metrics,
        dwi=dwi_metrics,
        training={
            "final_loss": final_loss,
            "best_loss": best_loss if math.isfinite(best_loss) else final_loss,
            "best_epoch": int(best_epoch),
            "training_time_sec": float(train_time),
            "epochs": int(epochs),
        },
        extra={
            "n_volumes": int(vol_m.sum()),
            "device": str(device),
            "n_brain_voxels": n_brain,
            "n_wls_valid_voxels": n_wls_valid,
            "n_common_voxels": n_common,
            "training_mask": "brain",
            "evaluation_mask": "brain & WLS_valid",
        },
    )

    ckpt_payload = {
        "model": model.state_dict(),
        "subject_id": sid,
        "experiment": "independent_inr",
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss) if math.isfinite(best_loss) else float(final_loss),
        "config": {
            "hidden": hidden,
            "layers": layers,
            "pe_freqs": pe_freqs,
            "epochs": epochs,
            "batch_voxels": batch_voxels,
            "lr": lr,
            "seed": seed,
        },
    }
    save_subject_outputs(
        out_dir=out_dir,
        sid=sid,
        model=model,
        maps=maps,
        metrics_obj=metrics_obj,
        ckpt_payload=ckpt_payload,
        affine=affine,
        save_nifti_flag=save_nifti_flag,
        save_tensor_flag=save_tensor_flag,
    )

    row = metrics_json_to_summary_row(metrics_obj, ok=True)
    print(
        f"[{tag}] {sid} done | FA_MAE={row['FA_MAE']:.4f} MD_MAE={row['MD_MAE']:.6f} "
        f"DWI_RelMSE={row['DWI_RelMSE']:.6e} best@{best_epoch} → {out_dir}"
    )

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row
