"""Shared INR + subject latent training and evaluation.

One shared network θ and one embedding z_s per subject, trained jointly.
Loss, physics, masks, and evaluation protocol match Independent INR.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .coords import masked_coords_and_indices
from .hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask
from .io_utils import save_json
from .metrics_schema import (
    build_metrics_json,
    dwi_reconstruction_metrics,
    metrics_json_to_summary_row,
    parameter_agreement_vs_wls,
    write_summary_and_aggregate,
)
from .physics import compute_fa_md_ad_rd, dti_forward_signal
from .shared_model import SharedSpatialDTIINR
from .train_independent import load_or_fit_wls_reference, resolve_device


@dataclass
class SubjectTrainingData:
    subject_id: str
    subject_idx: int
    shape_xyz: tuple[int, int, int]
    affine: np.ndarray
    brain_mask: np.ndarray
    common_mask: np.ndarray
    train_coords: np.ndarray
    train_flat_idx: np.ndarray
    eval_coords: np.ndarray
    eval_flat_idx: np.ndarray
    dwi_flat: np.ndarray
    bvals_u: np.ndarray
    bvecs_u: np.ndarray
    ref: dict[str, Any]
    n_volumes: int


def build_subject_mapping(subject_ids: list[str]) -> dict[str, int]:
    """Stable explicit mapping: subject_id → embedding index."""
    return {str(sid): int(i) for i, sid in enumerate(subject_ids)}


def save_subject_mapping(path: Path, mapping: dict[str, int]) -> None:
    save_json(path, dict(sorted(mapping.items(), key=lambda kv: kv[1])))


def load_subject_mapping(path: Path) -> dict[str, int]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"invalid subject mapping: {path}")
    return {str(k): int(v) for k, v in obj.items()}


def prepare_subject_data(
    *,
    subject_id: str,
    subject_idx: int,
    cfg: dict[str, Any],
    trad_dir: Path,
    skip_traditional_if_exists: bool = True,
    train_volume_indices: np.ndarray | None = None,
    sampling_fraction: float | None = None,
) -> SubjectTrainingData:
    sid = str(subject_id).strip()
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
    n_volumes_full = int(dwi.shape[-1])
    if train_volume_indices is not None:
        vi = np.asarray(train_volume_indices, dtype=np.int64)
        dwi = dwi[..., vi]
        bvals_u = bvals_u[vi]
        bvecs_u = bvecs_u[vi]

    ref = load_or_fit_wls_reference(
        bundle=bundle,
        trad_dir=Path(trad_dir),
        cfg=cfg,
        skip_if_exists=skip_traditional_if_exists,
    )

    train_coords, train_flat_idx = masked_coords_and_indices(brain)
    common_mask = brain & ref["valid_mask"]
    if int(np.count_nonzero(common_mask)) == 0:
        raise RuntimeError(f"{sid}: empty common_mask")
    eval_coords, eval_flat_idx = masked_coords_and_indices(common_mask)

    return SubjectTrainingData(
        subject_id=sid,
        subject_idx=int(subject_idx),
        shape_xyz=tuple(int(x) for x in data.shape[:3]),
        affine=np.asarray(affine),
        brain_mask=np.asarray(brain, dtype=bool),
        common_mask=np.asarray(common_mask, dtype=bool),
        train_coords=np.asarray(train_coords, dtype=np.float32),
        train_flat_idx=np.asarray(train_flat_idx, dtype=np.int64),
        eval_coords=np.asarray(eval_coords, dtype=np.float32),
        eval_flat_idx=np.asarray(eval_flat_idx, dtype=np.int64),
        dwi_flat=dwi.reshape(-1, dwi.shape[-1]),
        bvals_u=bvals_u,
        bvecs_u=bvecs_u,
        ref=ref,
        n_volumes=int(dwi.shape[-1]),
    )


@torch.no_grad()
def predict_maps_shared(
    model: SharedSpatialDTIINR,
    coords: torch.Tensor,
    flat_idx: np.ndarray,
    shape_xyz: tuple[int, int, int],
    subject_idx: int,
    device: torch.device,
    *,
    chunk: int = 65536,
) -> dict[str, np.ndarray]:
    model.eval()
    X, Y, Z = shape_xyz
    n = coords.shape[0]
    sub_t = torch.full((1,), int(subject_idx), dtype=torch.long, device=device)
    S0_vol = np.zeros((X * Y * Z,), dtype=np.float32)
    FA = np.zeros((X * Y * Z,), dtype=np.float32)
    MD = np.zeros((X * Y * Z,), dtype=np.float32)
    AD = np.zeros((X * Y * Z,), dtype=np.float32)
    RD = np.zeros((X * Y * Z,), dtype=np.float32)

    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        xyz = coords[sl].to(device)
        idx = flat_idx[sl]
        S0, D = model(xyz, sub_t)
        fa, md, ad, rd = compute_fa_md_ad_rd(D.detach().float().cpu())
        S0_vol[idx] = S0.detach().float().cpu().numpy()
        FA[idx] = fa.numpy()
        MD[idx] = md.numpy()
        AD[idx] = ad.numpy()
        RD[idx] = rd.numpy()

    return {
        "S0": S0_vol.reshape(X, Y, Z),
        "FA": FA.reshape(X, Y, Z),
        "MD": MD.reshape(X, Y, Z),
        "AD": AD.reshape(X, Y, Z),
        "RD": RD.reshape(X, Y, Z),
    }


@torch.no_grad()
def evaluate_dwi_shared(
    model: SharedSpatialDTIINR,
    coords: torch.Tensor,
    flat_idx: np.ndarray,
    dwi_flat: np.ndarray,
    bvals_t: torch.Tensor,
    bvecs_t: torch.Tensor,
    subject_idx: int,
    device: torch.device,
    *,
    max_voxels: int = 131072,
    seed: int = 42,
    evaluation_mask: str = "brain & WLS_valid",
) -> dict[str, float]:
    """Same RelMSE protocol as Independent INR (seed + max_voxels on common mask)."""
    model.eval()
    n_eval = int(coords.shape[0])
    rng = np.random.default_rng(int(seed))
    sel = np.arange(n_eval) if n_eval <= max_voxels else rng.choice(n_eval, size=max_voxels, replace=False)
    n_sampled = int(sel.size)
    xyz = coords[sel].to(device)
    target = torch.from_numpy(dwi_flat[flat_idx[sel]]).to(device)
    sub_t = torch.full((xyz.shape[0],), int(subject_idx), dtype=torch.long, device=device)
    S0, D = model(xyz, sub_t)
    pred = dti_forward_signal(S0, D, bvals_t, bvecs_t)
    out = dwi_reconstruction_metrics(pred.detach().float().cpu().numpy(), target.detach().float().cpu().numpy())
    out["evaluation_mask"] = str(evaluation_mask)
    out["max_voxels"] = float(max_voxels)
    out["seed"] = float(seed)
    out["n_eval_voxels"] = float(n_eval)
    out["n_sampled_voxels"] = float(n_sampled)
    return out


def evaluate_shared_subject(
    *,
    model: SharedSpatialDTIINR,
    subj: SubjectTrainingData,
    device: torch.device,
    eval_seed: int = 42,
) -> dict[str, Any]:
    bvals_t = torch.from_numpy(subj.bvals_u).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs_u).to(device)
    train_coords_t = torch.from_numpy(subj.train_coords)
    eval_coords_t = torch.from_numpy(subj.eval_coords)

    maps = predict_maps_shared(
        model,
        train_coords_t,
        subj.train_flat_idx,
        subj.shape_xyz,
        subj.subject_idx,
        device,
    )
    param_metrics = parameter_agreement_vs_wls(maps, subj.ref, subj.common_mask)
    dwi_metrics = evaluate_dwi_shared(
        model,
        eval_coords_t,
        subj.eval_flat_idx,
        subj.dwi_flat,
        bvals_t,
        bvecs_t,
        subj.subject_idx,
        device,
        seed=int(eval_seed),
        evaluation_mask="brain & WLS_valid",
    )
    return build_metrics_json(
        subject_id=subj.subject_id,
        parameter_metrics=param_metrics,
        dwi=dwi_metrics,
        training={"final_loss": None, "best_loss": None, "best_epoch": None, "training_time_sec": None, "epochs": None},
        extra={
            "subject_idx": int(subj.subject_idx),
            "n_volumes": int(subj.n_volumes),
            "n_brain_voxels": int(np.count_nonzero(subj.brain_mask)),
            "n_wls_valid_voxels": int(np.count_nonzero(subj.ref["valid_mask"])),
            "n_common_voxels": int(np.count_nonzero(subj.common_mask)),
            "training_mask": "brain",
            "evaluation_mask": "brain & WLS_valid",
            "experiment": "shared_inr",
        },
    )


def _train_subject_batches(
    model: SharedSpatialDTIINR,
    opt: torch.optim.Optimizer,
    subj: SubjectTrainingData,
    *,
    device: torch.device,
    batch_voxels: int,
    b0_threshold: float,
    rng: np.random.Generator,
    train_coords_t: torch.Tensor,
    bvals_t: torch.Tensor,
    bvecs_t: torch.Tensor,
) -> float:
    model.train()
    n_vox = int(subj.train_coords.shape[0])
    n_steps = max(1, int(np.ceil(n_vox / batch_voxels)))
    sub_idx_t = torch.full((batch_voxels,), int(subj.subject_idx), dtype=torch.long, device=device)
    losses: list[float] = []
    for _ in range(n_steps):
        sel = rng.integers(0, n_vox, size=batch_voxels, endpoint=False)
        xyz = train_coords_t[sel].to(device)
        idx = subj.train_flat_idx[sel]
        target = torch.from_numpy(subj.dwi_flat[idx]).to(device)

        S0, D = model(xyz, sub_idx_t)
        pred = dti_forward_signal(S0, D, bvals_t, bvecs_t)
        s0_obs = target[:, bvals_t < float(b0_threshold)].mean(dim=-1, keepdim=True).clamp_min(1.0)
        loss = torch.mean(((pred - target) / s0_obs) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_shared_inr(
    *,
    subject_ids: list[str],
    cfg: dict[str, Any],
    out_root: Path,
    trad_root: Path,
    device: torch.device,
    latent_dim: int = 32,
    epochs: int = 200,
    batch_voxels: int = 4096,
    lr: float = 1e-3,
    hidden: int = 128,
    layers: int = 4,
    pe_freqs: int = 8,
    log_every: int = 10,
    seed: int = 42,
    skip_traditional_if_exists: bool = True,
    save_maps: bool = True,
    tag: str = "SharedINR",
    train_volume_indices: np.ndarray | None = None,
    sampling_fraction: float | None = None,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    maps_root = out_root / "maps"
    if save_maps:
        maps_root.mkdir(parents=True, exist_ok=True)

    mapping = build_subject_mapping(subject_ids)
    save_subject_mapping(out_root / "subject_mapping.json", mapping)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[{tag}] loading {len(subject_ids)} subjects …")
    subjects: list[SubjectTrainingData] = []
    for sid in subject_ids:
        subj = prepare_subject_data(
            subject_id=sid,
            subject_idx=mapping[sid],
            cfg=cfg,
            trad_dir=trad_root / sid,
            skip_traditional_if_exists=skip_traditional_if_exists,
            train_volume_indices=train_volume_indices,
            sampling_fraction=sampling_fraction,
        )
        subjects.append(subj)
        print(
            f"  [{sid}] idx={subj.subject_idx} brain={subj.train_coords.shape[0]} "
            f"common={subj.eval_coords.shape[0]} vols={subj.n_volumes}"
        )

    model = SharedSpatialDTIINR(
        num_subjects=len(subject_ids),
        latent_dim=int(latent_dim),
        hidden=int(hidden),
        layers=int(layers),
        pe_freqs=int(pe_freqs),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    # Per-subject cached tensors on CPU; bvals/bvecs moved per subject batch.
    coords_cache = {s.subject_id: torch.from_numpy(s.train_coords) for s in subjects}

    t0 = time.time()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_loss = float("nan")

    for epoch in range(1, int(epochs) + 1):
        order = list(subjects)
        rng.shuffle(order)
        per_subject_losses: list[float] = []
        for subj in order:
            bvals_t = torch.from_numpy(subj.bvals_u).to(device)
            bvecs_t = torch.from_numpy(subj.bvecs_u).to(device)
            ls = _train_subject_batches(
                model,
                opt,
                subj,
                device=device,
                batch_voxels=int(batch_voxels),
                b0_threshold=float(cfg["b0_threshold"]),
                rng=rng,
                train_coords_t=coords_cache[subj.subject_id],
                bvals_t=bvals_t,
                bvecs_t=bvecs_t,
            )
            per_subject_losses.append(ls)

        mean_loss = float(np.mean(per_subject_losses)) if per_subject_losses else float("nan")
        final_loss = mean_loss
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % int(log_every) == 0 or epoch == 1 or epoch == int(epochs):
            print(
                f"  [{tag}] epoch {epoch:4d}/{epochs}  L_DWI_mean={mean_loss:.6e}  "
                f"best={best_loss:.6e}@{best_epoch}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    train_cfg = {
        "latent_dim": int(latent_dim),
        "hidden": int(hidden),
        "layers": int(layers),
        "pe_freqs": int(pe_freqs),
        "epochs": int(epochs),
        "batch_voxels": int(batch_voxels),
        "lr": float(lr),
        "seed": int(seed),
        "num_subjects": len(subject_ids),
        "sampling_fraction": sampling_fraction,
        "n_train_volumes": int(train_volume_indices.size) if train_volume_indices is not None else None,
    }
    ckpt = {
        "model": model.state_dict(),
        "subject_embedding": model.subject_embedding.state_dict(),
        "subject_mapping": mapping,
        "experiment": "shared_inr",
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss) if math.isfinite(best_loss) else float(final_loss),
        "final_loss": float(final_loss),
        "training_time_sec": float(train_time),
        "config": train_cfg,
    }
    torch.save(ckpt, out_root / "best.pt")

    (out_root / "metrics").mkdir(parents=True, exist_ok=True)

    with open(out_root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "shared_inr": train_cfg,
                "subjects": subject_ids,
                "subject_mapping": mapping,
                "evaluation_mask": "brain & WLS_valid",
                "eval_seed": int(seed),
                "max_voxels": 131072,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"[{tag}] evaluating {len(subjects)} subjects …")
    rows: list[dict[str, Any]] = []
    metrics_by_sid: dict[str, dict[str, Any]] = {}
    for subj in subjects:
        metrics_obj = evaluate_shared_subject(model=model, subj=subj, device=device, eval_seed=int(seed))
        metrics_obj.setdefault("extra", {})
        metrics_obj["extra"]["sampling_fraction"] = sampling_fraction
        metrics_obj["training"] = {
            "final_loss": final_loss,
            "best_loss": float(best_loss) if math.isfinite(best_loss) else final_loss,
            "best_epoch": int(best_epoch),
            "training_time_sec": float(train_time),
            "epochs": int(epochs),
        }
        metrics_by_sid[subj.subject_id] = metrics_obj
        save_json(out_root / "metrics" / f"{subj.subject_id}.json", metrics_obj)
        if save_maps:
            maps = predict_maps_shared(
                model,
                coords_cache[subj.subject_id],
                subj.train_flat_idx,
                subj.shape_xyz,
                subj.subject_idx,
                device,
            )
            np.savez_compressed(
                maps_root / f"{subj.subject_id}.npz",
                S0=maps["S0"].astype(np.float32),
                FA=maps["FA"].astype(np.float32),
                MD=maps["MD"].astype(np.float32),
                AD=maps["AD"].astype(np.float32),
                RD=maps["RD"].astype(np.float32),
            )
        row = metrics_json_to_summary_row(metrics_obj, ok=True)
        rows.append(row)
        print(
            f"  [{subj.subject_id}] FA_MAE={row['FA_MAE']:.4f} DWI_RelMSE={row['DWI_RelMSE']:.6e}"
        )

    write_summary_and_aggregate(out_root, rows)
    write_comparison_table(
        out_root=out_root,
        shared_rows=rows,
        independent_summary=Path(cfg.get("_independent_summary_path", "")),
    )
    write_shared_aggregate_report(out_root, rows, mapping)

    del model, opt, subjects
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "out_root": str(out_root),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "training_time_sec": float(train_time),
        "n_subjects": len(rows),
        "rows": rows,
    }


def load_shared_checkpoint(path: Path, device: torch.device) -> tuple[SharedSpatialDTIINR, dict[str, int], dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    mapping = ckpt.get("subject_mapping") or {}
    if not mapping and (Path(path).parent / "subject_mapping.json").is_file():
        mapping = load_subject_mapping(Path(path).parent / "subject_mapping.json")
    model = SharedSpatialDTIINR(
        num_subjects=len(mapping),
        latent_dim=int(c.get("latent_dim", 32)),
        hidden=int(c.get("hidden", 128)),
        layers=int(c.get("layers", 4)),
        pe_freqs=int(c.get("pe_freqs", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mapping, ckpt


def write_comparison_table(
    *,
    out_root: Path,
    shared_rows: list[dict[str, Any]],
    independent_summary: Path,
) -> None:
    import csv

    out_root = Path(out_root)
    indep_path = Path(independent_summary)
    if not indep_path.is_file():
        print(f"[SharedINR] skip comparison: independent summary not found at {indep_path}")
        return

    indep = {}
    with open(indep_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            indep[r["subject_id"]] = r

    fields = [
        "subject_id",
        "Independent_DWI_RelMSE",
        "Shared_DWI_RelMSE",
        "DWI_delta",
        "Independent_FA_MAE",
        "Shared_FA_MAE",
        "FA_delta",
        "Independent_MD_MAE",
        "Shared_MD_MAE",
        "MD_delta",
        "Independent_AD_MAE",
        "Shared_AD_MAE",
        "AD_delta",
        "Independent_RD_MAE",
        "Shared_RD_MAE",
        "RD_delta",
    ]
    rows = []
    for sr in shared_rows:
        sid = str(sr["subject_id"])
        ir = indep.get(sid, {})
        if not ir:
            continue

        def _f(x: Any) -> float:
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("nan")

        ind_dwi = _f(ir.get("DWI_RelMSE"))
        sh_dwi = _f(sr.get("DWI_RelMSE"))
        ind_fa = _f(ir.get("FA_MAE"))
        sh_fa = _f(sr.get("FA_MAE"))
        ind_md = _f(ir.get("MD_MAE"))
        sh_md = _f(sr.get("MD_MAE"))
        ind_ad = _f(ir.get("AD_MAE"))
        sh_ad = _f(sr.get("AD_MAE"))
        ind_rd = _f(ir.get("RD_MAE"))
        sh_rd = _f(sr.get("RD_MAE"))
        rows.append(
            {
                "subject_id": sid,
                "Independent_DWI_RelMSE": ind_dwi,
                "Shared_DWI_RelMSE": sh_dwi,
                "DWI_delta": sh_dwi - ind_dwi,
                "Independent_FA_MAE": ind_fa,
                "Shared_FA_MAE": sh_fa,
                "FA_delta": sh_fa - ind_fa,
                "Independent_MD_MAE": ind_md,
                "Shared_MD_MAE": sh_md,
                "MD_delta": sh_md - ind_md,
                "Independent_AD_MAE": ind_ad,
                "Shared_AD_MAE": sh_ad,
                "AD_delta": sh_ad - ind_ad,
                "Independent_RD_MAE": ind_rd,
                "Shared_RD_MAE": sh_rd,
                "RD_delta": sh_rd - ind_rd,
            }
        )

    cmp_path = out_root / "comparison_independent_vs_shared.csv"
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[SharedINR] wrote {cmp_path}")


def write_shared_aggregate_report(
    out_root: Path,
    rows: list[dict[str, Any]],
    mapping: dict[str, int],
) -> None:
    """Extend aggregate.md with MVP questions and focus-subject section."""
    base = (out_root / "aggregate.md").read_text(encoding="utf-8") if (out_root / "aggregate.md").is_file() else ""
    focus = ("112920", "124422", "130720", "101309")

    def _vals(key: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(key)
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out.append(fv)
            except (TypeError, ValueError):
                pass
        return out

    def _stat(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        a = np.asarray(vals, dtype=np.float64)
        return f"mean={a.mean():.6g}, median={np.median(a):.6g}, std={a.std():.6g}"

    dwi = _vals("DWI_RelMSE")
    fa = _vals("FA_MAE")
    by_sid = {str(r["subject_id"]): r for r in rows}

    lines = [
        base.rstrip(),
        "",
        "## Shared INR MVP notes",
        "",
        f"- One shared network + `{len(mapping)}` subject embeddings (latent table)",
        "- Training: all subjects update the same θ and their z_s each epoch",
        "- Evaluation: same `brain & WLS_valid`, seed=42, max_voxels=131072 as Independent INR",
        "",
        "### Question 1 — training stability",
        "- Check `best.pt`, `summary.csv`, and per-epoch logs for NaN / divergence.",
        "",
        "### Question 2 — DWI reconstruction vs Independent",
        f"- Shared DWI RelMSE: {_stat(dwi)}",
        "- See `comparison_independent_vs_shared.csv` for per-subject Δ (Shared − Independent).",
        "",
        "### Question 3 — subject-to-subject variability",
        f"- Shared FA MAE spread: {_stat(fa)}",
        "",
        "### Question 4 — prior Independent failure subjects",
        "",
        "| subject | Shared DWI RelMSE | Shared FA MAE | note |",
        "|---------|------------------:|--------------:|------|",
    ]
    for sid in ("112920", "124422", "130720"):
        r = by_sid.get(sid)
        if r:
            lines.append(
                f"| {sid} | {float(r['DWI_RelMSE']):.6g} | {float(r['FA_MAE']):.4g} | "
                f"compare Δ in comparison CSV |"
            )
    lines += [
        "",
        "### Question 5 — new failures",
        "- Subjects with highest Shared FA MAE or DWI RelMSE vs cohort median should be reviewed manually.",
        "",
        "### Focus subjects (not hard-coded failures)",
        "",
    ]
    for sid in focus:
        r = by_sid.get(sid)
        if not r:
            continue
        lines.append(
            f"- **{sid}**: DWI_RelMSE={float(r['DWI_RelMSE']):.6g}, FA_MAE={float(r['FA_MAE']):.4g}, "
            f"MD_MAE={float(r['MD_MAE']):.6g}"
        )
    lines.append("")
    (out_root / "aggregate.md").write_text("\n".join(lines), encoding="utf-8")
