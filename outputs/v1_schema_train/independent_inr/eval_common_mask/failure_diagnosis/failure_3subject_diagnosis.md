# Failure diagnosis: 112920 / 124422 / 130720 (+ control 103515)

Protocol: `common_mask = brain & WLS_valid`; DWI RelMSE from G2 shared voxels (seed=42, n=131072).

## Compact table

| subject | role | N_brain | N_valid | N_common | raw_b0_mean | S0_hat_p99 | S0_max | frac_S0>1e5 | WLS_RelMSE | INR_RelMSE | FA_MAE | MD_MAE | INR_FA_mean | WLS_FA_mean | case |
|---------|------|--------:|--------:|---------:|------------:|-----------:|-------:|------------:|-----------:|-----------:|-------:|-------:|------------:|------------:|------|
| 112920 | failure | 948104 | 947861 | 947861 | 4058.9 | 12155.3 | 937491.1 | 7.70e-05 | 0.3580 | 0.1816 | 0.462 | 0.000419 | 0.716 | 0.259 | INR_optimization_collapse |
| 124422 | failure | 731536 | 731407 | 731407 | 4864.0 | 16004.4 | 988862.8 | 8.20e-05 | 0.2472 | 0.1877 | 0.465 | 0.000425 | 0.716 | 0.256 | INR_reconstruction_and_FA_failure |
| 130720 | failure | 856215 | 856089 | 856089 | 4427.8 | 14685.7 | 948661.6 | 4.67e-05 | 0.2414 | 0.1241 | 0.488 | 0.002244 | 0.727 | 0.242 | INR_failure_eigenvalue_blowup |
| 103515 | control | 748637 | 748489 | 748489 | 3896.3 | 10686.6 | 839015.4 | 6.41e-05 | 0.4328 | 0.0494 | 0.146 | 0.000169 | 0.151 | 0.261 | control_or_stable |

## How to read

1. **Raw data hard?** Compare `raw_b0_mean` / `raw_b1000_*` and mask sizes to control. If similar → not obvious raw-data catastrophe.
2. **WLS itself fails?** Global `WLS_DWI_RelMSE` under `valid_mask` (S0<1e6) is often dominated by rare `S0_hat` explosions (`frac_S0>1e5`, `S0_max`). High WLS RelMSE ≠ WLS cannot fit typical tissue.
3. **INR truly fails?** Look at `INR_DWI_RelMSE` (~0.12–0.19 vs control ~0.05) and especially `FA_MAE` (~0.46) with `INR_FA_mean` collapsed vs `WLS_FA_mean`.

## Verdict template

- If mask counts & raw intensity ≈ control, but FA_MAE≫0.3 and INR RelMSE elevated → **INR failure**.
- If S0 tails huge and WLS RelMSE huge but FA maps look normal and INR FA collapsed → **INR param failure**; do not blame raw data solely from WLS RelMSE.
