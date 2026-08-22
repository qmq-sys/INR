# Experiment 3 — Low DWI Sampling (Independent vs Shared)

- Protocol seed: **42**
- Reference subject for volume indices: **101309**
- Training: nested volume subsets (b0 + b1000); spatial training mask = brain
- Evaluation: common_mask = brain & WLS_valid; eval seed=42; max_voxels=131072
- 100% results reused from frozen Independent + Shared MVP (no retrain)

## Cohort medians

| level | n_vol | Independent DWI | Shared DWI | Δ DWI | Independent FA | Shared FA | Δ FA | Shared better DWI | Shared better FA |
|-------|------:|----------------:|-----------:|------:|---------------:|----------:|-----:|------------------:|-----------------:|
| 100% | 108 | 0.0705 | 0.1464 | +0.05686 | 0.1595 | 0.1914 | +0.03 | 2/29 | 5/29 |
| 50% | 54 | 1 | 1 | -6.084e-11 | 0.1641 | 0.1624 | -0.001703 | 1/1 | 1/1 |

## Focus subjects

### 112920

| level | Ind DWI | Sh DWI | Δ DWI | Ind FA | Sh FA | Δ FA |
|-------|--------:|-------:|------:|-------:|------:|-----:|
| 100% | 0.1816 | 0.1206 | -0.061 | 0.4621 | 0.1988 | -0.2634 |

### 124422

| level | Ind DWI | Sh DWI | Δ DWI | Ind FA | Sh FA | Δ FA |
|-------|--------:|-------:|------:|-------:|------:|-----:|
| 100% | 0.1877 | 0.1727 | -0.01505 | 0.4654 | 0.1982 | -0.2673 |

### 130720

| level | Ind DWI | Sh DWI | Δ DWI | Ind FA | Sh FA | Δ FA |
|-------|--------:|-------:|------:|-------:|------:|-----:|
| 100% | 0.1241 | 0.1548 | +0.03078 | 0.4885 | 0.1852 | -0.3033 |

### 101309

| level | Ind DWI | Sh DWI | Δ DWI | Ind FA | Sh FA | Δ FA |
|-------|--------:|-------:|------:|-------:|------:|-----:|
| 100% | 0.06364 | 0.08361 | +0.01998 | 0.2243 | 0.217 | -0.007334 |
| 50% | 1 | 1 | -6.084e-11 | 0.1641 | 0.1624 | -0.001703 |

## Interpretation (candidate only)

- Look for **crossover**: Shared curve degrades slower as sampling drops.
- Negative Δ DWI / Δ FA means Shared better than Independent at that subject/level.
- No automatic claim of superiority — inspect curves in `figures/`.
