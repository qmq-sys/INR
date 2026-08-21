# Common-mask DWI re-evaluation (Phase 1)

## Common mask definition (model-independent)

```text
common_dwi_eval_mask =
    brain_mask
    AND WLS_valid_mask   # from inr/dti_fit.py valid_mask (unchanged)
    AND finite(observed_DWI)   # all volumes finite at the voxel
```

- Prediction finiteness (`finite(WLS_pred)` / `finite(INR_pred)`) is **not** in the common mask.
- If a prediction is nonfinite on the shared eval set, RelMSE is set to **NaN** (no silent voxel drop).
- WLS and INR use the **same** deterministic `eval_voxel_indices` (seed=42, cap=131072).
- `best.pt` still means **minimum training-loss** checkpoint; this phase does not retrain.
- `best_DWI_epoch` / `best_DWI_RelMSE` are **not recoverable** from existing runs → recorded as empty / N/A.
- No `1 < S0 < 15000` filter is used.

## Aggregate

- N subjects: 29
- WLS RelMSE_common median: 0.24718
- INR RelMSE_common median: 0.070499
- Total WLS nonfinite entries (all subjects): 0
- Total INR nonfinite entries (all subjects): 0

## Per-subject table

| subject | n_common | n_eval | WLS RelMSE | INR RelMSE | legacy INR RelMSE | n_wls_nf | n_inr_nf | FA_MAE |
|---------|---------:|-------:|-----------:|-----------:|------------------:|---------:|---------:|-------:|
| 130720 ** | 856089 | 131072 | 0.241434 | 0.124067 | 0.124844 | 0 | 0 | 0.4885 |
| 124422 ** | 731407 | 131072 | 0.24718 | 0.187722 | 0.18885 | 0 | 0 | 0.4654 |
| 112920 ** | 947861 | 131072 | 0.358047 | 0.18156 | 0.182363 | 0 | 0 | 0.4621 |
| 101309 | 799510 | 131072 | 0.163615 | 0.0636366 | 0.063695 | 0 | 0 | 0.2243 |
| 114419 | 923360 | 131072 | 0.475522 | 0.106659 | 0.106378 | 0 | 0 | 0.1979 |
| 102715 | 940977 | 131072 | 0.0176203 | 0.10384 | 0.104999 | 0 | 0 | 0.1910 |
| 124624 | 791245 | 131072 | 0.270107 | 0.102706 | 0.104018 | 0 | 0 | 0.1803 |
| 107725 | 751406 | 131072 | 0.198566 | 0.0553231 | 0.0560999 | 0 | 0 | 0.1790 |
| 131217 | 751792 | 131072 | 0.254602 | 0.0782379 | 0.0784166 | 0 | 0 | 0.1755 |
| 114116 | 703523 | 131072 | 0.677461 | 0.0868533 | 0.0875325 | 0 | 0 | 0.1662 |
| 107321 | 716567 | 131072 | 0.430686 | 0.0698918 | 0.0696981 | 0 | 0 | 0.1651 |
| 121618 | 826775 | 131072 | 0.101895 | 0.0773498 | 0.0777512 | 0 | 0 | 0.1639 |
| 130316 | 787734 | 131072 | 0.0839871 | 0.070499 | 0.0713461 | 0 | 0 | 0.1629 |
| 108525 | 767701 | 131072 | 0.574203 | 0.0760478 | 0.0757681 | 0 | 0 | 0.1628 |
| 122418 | 766180 | 131072 | 0.161044 | 0.0472437 | 0.0473621 | 0 | 0 | 0.1595 |
| 115825 | 780336 | 131072 | 0.042713 | 0.0702321 | 0.0709627 | 0 | 0 | 0.1588 |
| 106319 | 793174 | 131072 | 0.492064 | 0.091143 | 0.0913028 | 0 | 0 | 0.1583 |
| 111211 | 737449 | 131072 | 0.0888829 | 0.058572 | 0.0586125 | 0 | 0 | 0.1572 |
| 130518 | 679959 | 131072 | 0.978262 | 0.0782006 | 0.0791241 | 0 | 0 | 0.1564 |
| 110613 | 878568 | 131072 | 0.204496 | 0.0750967 | 0.0751524 | 0 | 0 | 0.1553 |
| 104416 | 750597 | 131072 | 0.228638 | 0.0652626 | 0.0647847 | 0 | 0 | 0.1518 |
| 109830 | 633273 | 131072 | 0.106982 | 0.0548996 | 0.0546054 | 0 | 0 | 0.1508 |
| 120717 | 784248 | 131072 | 0.313476 | 0.0555576 | 0.0555703 | 0 | 0 | 0.1497 |
| 130922 | 767727 | 131072 | 0.568681 | 0.0548631 | 0.0546882 | 0 | 0 | 0.1491 |
| 107422 | 768674 | 131072 | 0.255967 | 0.0576593 | 0.0574382 | 0 | 0 | 0.1482 |
| 129937 | 757597 | 131072 | 0.634046 | 0.0675749 | 0.0682384 | 0 | 0 | 0.1481 |
| 116726 | 786932 | 131072 | 0.0965027 | 0.0847024 | 0.0849174 | 0 | 0 | 0.1480 |
| 123925 | 685494 | 131072 | 0.0767618 | 0.0626257 | 0.0628637 | 0 | 0 | 0.1471 |
| 103515 | 748489 | 131072 | 0.432776 | 0.0493883 | 0.049237 | 0 | 0 | 0.1457 |

## Known failure subjects

### 112920

- n_common_dwi_voxels: 947861
- n_eval_dwi_voxels: 131072
- n_common_dwi_values: 14155776
- WLS_DWI_RelMSE_common: 0.35804742771676074
- INR_DWI_RelMSE_common: 0.18155966469660242
- legacy_INR_DWI_RelMSE (brain-only protocol): 0.18236288553591362
- n_wls_nonfinite: 0
- n_inr_nonfinite: 0
- FA_MAE (unchanged vs WLS maps): 0.4621490325545107
- best_loss / best_loss_epoch: 0.43829957007205694 / 57
- best_DWI_*: N/A for existing checkpoints (eval_every unused; not logged historically)

### 124422

- n_common_dwi_voxels: 731407
- n_eval_dwi_voxels: 131072
- n_common_dwi_values: 14155776
- WLS_DWI_RelMSE_common: 0.24718025184282935
- INR_DWI_RelMSE_common: 0.1877224648701143
- legacy_INR_DWI_RelMSE (brain-only protocol): 0.1888502259029822
- n_wls_nonfinite: 0
- n_inr_nonfinite: 0
- FA_MAE (unchanged vs WLS maps): 0.4654457809193726
- best_loss / best_loss_epoch: 0.1160451283251773 / 73
- best_DWI_*: N/A for existing checkpoints (eval_every unused; not logged historically)

### 130720

- n_common_dwi_voxels: 856089
- n_eval_dwi_voxels: 131072
- n_common_dwi_values: 14155776
- WLS_DWI_RelMSE_common: 0.24143402315441867
- INR_DWI_RelMSE_common: 0.12406725822286387
- legacy_INR_DWI_RelMSE (brain-only protocol): 0.12484396547535777
- n_wls_nonfinite: 0
- n_inr_nonfinite: 0
- FA_MAE (unchanged vs WLS maps): 0.488457583603139
- best_loss / best_loss_epoch: 0.15841551009742988 / 192
- best_DWI_*: N/A for existing checkpoints (eval_every unused; not logged historically)

