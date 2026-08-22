# Experiment aggregate (mean ± std)

- N subjects (ok): **1**
- Parameter MAE/RMSE/r = agreement vs WLS reference (not GT error)

| Metric | mean ± std | median |
|--------|-----------:|-------:|
| FA_MAE | 0.162361 ± 0 | 0.162361 |
| FA_RMSE | 0.196876 ± 0 | 0.196876 |
| FA_r | -0.0214779 ± 0 | -0.0214779 |
| MD_MAE | 0.0507131 ± 0 | 0.0507131 |
| MD_RMSE | 0.0507149 ± 0 | 0.0507149 |
| MD_r | 0.0299244 ± 0 | 0.0299244 |
| AD_MAE | 0.0612777 ± 0 | 0.0612777 |
| AD_RMSE | 0.0612792 ± 0 | 0.0612792 |
| AD_r | -0.0248911 ± 0 | -0.0248911 |
| RD_MAE | 0.0454308 ± 0 | 0.0454308 |
| RD_RMSE | 0.0454335 ± 0 | 0.0454335 |
| RD_r | 0.0307366 ± 0 | 0.0307366 |
| DWI_MAE | 2097.29 ± 0 | 2097.29 |
| DWI_RelMSE | 0.999999 ± 0 | 0.999999 |
| final_loss | 0.552079 ± 0 | 0.552079 |
| best_loss | 0.552079 ± 0 | 0.552079 |
| training_time_sec | 3.15596 ± 0 | 3.15596 |

## Shared INR MVP notes

- One shared network + `1` subject embeddings (latent table)
- Training: all subjects update the same θ and their z_s each epoch
- Evaluation: same `brain & WLS_valid`, seed=42, max_voxels=131072 as Independent INR

### Question 1 — training stability
- Check `best.pt`, `summary.csv`, and per-epoch logs for NaN / divergence.

### Question 2 — DWI reconstruction vs Independent
- Shared DWI RelMSE: mean=0.999999, median=0.999999, std=0
- See `comparison_independent_vs_shared.csv` for per-subject Δ (Shared − Independent).

### Question 3 — subject-to-subject variability
- Shared FA MAE spread: mean=0.162361, median=0.162361, std=0

### Question 4 — prior Independent failure subjects

| subject | Shared DWI RelMSE | Shared FA MAE | note |
|---------|------------------:|--------------:|------|

### Question 5 — new failures
- Subjects with highest Shared FA MAE or DWI RelMSE vs cohort median should be reviewed manually.

### Focus subjects (not hard-coded failures)

- **101309**: DWI_RelMSE=0.999999, FA_MAE=0.1624, MD_MAE=0.0507131
