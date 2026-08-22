# Experiment aggregate (mean ± std)

- N subjects (ok): **29**
- Parameter MAE/RMSE/r = agreement vs WLS reference (not GT error)

| Metric | mean ± std | median |
|--------|-----------:|-------:|
| FA_MAE | 0.193748 ± 0.0091006 | 0.191384 |
| FA_RMSE | 0.261946 ± 0.00977037 | 0.260418 |
| FA_r | 0.15262 ± 0.0326988 | 0.1508 |
| MD_MAE | 0.000279585 ± 3.6504e-05 | 0.000281507 |
| MD_RMSE | 0.000502298 ± 5.72636e-05 | 0.000504538 |
| MD_r | 0.345418 ± 0.0388136 | 0.35483 |
| AD_MAE | 0.000384711 ± 3.69045e-05 | 0.000387709 |
| AD_RMSE | 0.000590996 ± 5.40847e-05 | 0.000593235 |
| AD_r | 0.344042 ± 0.039679 | 0.351457 |
| RD_MAE | 0.000308666 ± 3.31674e-05 | 0.000310852 |
| RD_RMSE | 0.000497674 ± 5.36873e-05 | 0.000498323 |
| RD_r | 0.335381 ± 0.0385396 | 0.342406 |
| DWI_MAE | 520.225 ± 72.2649 | 496.555 |
| DWI_RelMSE | 0.138824 ± 0.0274098 | 0.146415 |
| final_loss | 0.583614 ± 0 | 0.583614 |
| best_loss | 0.4498 ± 0 | 0.4498 |
| training_time_sec | 5941.31 ± 0 | 5941.31 |

## Shared INR MVP notes

- One shared network + `29` subject embeddings (latent table)
- Training: all subjects update the same θ and their z_s each epoch
- Evaluation: same `brain & WLS_valid`, seed=42, max_voxels=131072 as Independent INR

### Question 1 — training stability
- Check `best.pt`, `summary.csv`, and per-epoch logs for NaN / divergence.

### Question 2 — DWI reconstruction vs Independent
- Shared DWI RelMSE: mean=0.138824, median=0.146415, std=0.0269331
- See `comparison_independent_vs_shared.csv` for per-subject Δ (Shared − Independent).

### Question 3 — subject-to-subject variability
- Shared FA MAE spread: mean=0.193748, median=0.191384, std=0.00894232

### Question 4 — prior Independent failure subjects

| subject | Shared DWI RelMSE | Shared FA MAE | note |
|---------|------------------:|--------------:|------|
| 112920 | 0.12056 | 0.1988 | compare Δ in comparison CSV |
| 124422 | 0.172671 | 0.1982 | compare Δ in comparison CSV |
| 130720 | 0.154846 | 0.1852 | compare Δ in comparison CSV |

### Question 5 — new failures
- Subjects with highest Shared FA MAE or DWI RelMSE vs cohort median should be reviewed manually.

### Focus subjects (not hard-coded failures)

- **112920**: DWI_RelMSE=0.12056, FA_MAE=0.1988, MD_MAE=0.000243711
- **124422**: DWI_RelMSE=0.172671, FA_MAE=0.1982, MD_MAE=0.000316447
- **130720**: DWI_RelMSE=0.154846, FA_MAE=0.1852, MD_MAE=0.00030612
- **101309**: DWI_RelMSE=0.083613, FA_MAE=0.217, MD_MAE=0.000209221
