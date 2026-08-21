# Failure Diagnosis Aggregate (Independent INR, 29 subjects)

## Protocol

- common_mask = `brain & WLS_valid` (reuse `valid_mask` / `s0_ok` from existing WLS fit; no new S0 threshold)
- shared sampled voxels: seed=42, max_voxels=131072
- WLS and INR DWI RelMSE use identical indices via `evaluate_wls_inr_dwi_common` / G2 reuse
- RelMSE = Σ(pred−obs)² / Σ(obs)²
- **No model / training / dti_fit changes** — diagnosis only

## Outlier rule (not a fixed absolute threshold)

- Upper fence: `x > Q3 + 1.5·IQR` on the 29-subject cohort
- OR elevated if robust z ≥ 2.5 (robust z = `(x − median) / (1.4826 · MAD)`)
- `WLS_FA_mean` abnormal: outside `[Q1−k·IQR, Q3+k·IQR]`
- Note: high `WLS_DWI_RelMSE` alone can reflect rare exploding `S0_hat` under `S0 < 1e6`; classification therefore also checks WLS FA mean fences and INR FA MAE.

## Overall

- N subjects = 29

### WLS DWI RelMSE
- mean=0.302628  median=0.24718  std=0.223273
- Q1=0.106982  Q3=0.432776  IQR=0.325795

### INR DWI RelMSE
- mean=0.0812902  median=0.070499  std=0.0334863
- Q1=0.058572  Q3=0.0868533  IQR=0.0282813

### INR/WLS ratio
- mean=0.610872  median=0.307295

### INR FA MAE (vs WLS)
- mean=0.195483  median=0.159471  std=0.0955621

## Status counts

- normal: 25
- inr_specific: 3
- parameter_only: 1
- wls_difficult: 0
- data_or_wls_suspect: 0

## Failure candidates (auto + focus)

| subject | status | WLS RelMSE | INR RelMSE | FA MAE | WLS FA mean | INR FA mean | reason |
|---------|--------|-----------:|-----------:|-------:|------------:|------------:|--------|
| 101309 | parameter_only | 0.1636 | 0.0636 | 0.2243 | 0.2697 | 0.1170 | FA MAE IQR-high but INR DWI not; reconstruction OK-ish, parameter agreement weak |
| 112920 | inr_specific | 0.3580 | 0.1816 | 0.4621 | 0.2592 | 0.7159 | INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure |
| 124422 | inr_specific | 0.2472 | 0.1877 | 0.4654 | 0.2555 | 0.7161 | INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure |
| 130720 | inr_specific | 0.2414 | 0.1241 | 0.4885 | 0.2422 | 0.7274 | INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure |

## Focus subjects (112920 / 124422 / 130720)

- **112920**: status=`inr_specific` — INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure (INR RelMSE=0.1816, FA MAE=0.4621, robust_z_INR_DWI=5.27, robust_z_FA_MAE=19.64)
- **124422**: status=`inr_specific` — INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure (INR RelMSE=0.1877, FA MAE=0.4654, robust_z_INR_DWI=5.57, robust_z_FA_MAE=19.85)
- **130720**: status=`inr_specific` — INR DWI and FA MAE IQR-high while WLS FA mean within cohort fences; consistent with INR-specific failure (INR RelMSE=0.1241, FA MAE=0.4885, robust_z_INR_DWI=2.54, robust_z_FA_MAE=21.35)

## Interpretation (candidates only)

- **subject/data difficulty**: raw intensity / volume counts far from cohort, or mixed WLS+INR fences.
- **WLS difficulty**: WLS DWI + WLS FA mean fences without INR DWI fence.
- **INR-specific failure**: INR DWI / FA MAE fences while WLS FA mean stays in cohort.
- **parameter-only failure**: FA MAE fence without INR DWI fence.

Do not treat this table as a final paper claim — use it to decide whether Independent INR can freeze.
