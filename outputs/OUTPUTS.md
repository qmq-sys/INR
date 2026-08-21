# INR outputs layout

Do **not** mix these three Independent-INR trees.

| Directory | What it is | Trained after new metrics? |
|-----------|------------|----------------------------|
| `v0_preschema/independent_inr/` | Original 29× Independent INR **training** (old file dump: NIfTI, `checkpoint.pt`, old metrics) | **No** |
| `v1_schema_reeval/independent_inr/` | Same v0 weights, **re-scored** with the fixed schema (`best.pt` / `maps.npz` / `metrics.json` + `summary.csv`) | **No** (eval only) |
| `v1_schema_train/independent_inr/` | Future Independent INR **training** that uses the new trainer + schema | Only after you run `step3_independent_inr.py` |

Single-subject:

| Directory | What it is |
|-----------|------------|
| `v0_preschema/single_inr/` | Original Exp1 run for 101309 |
| `v1_schema_train/single_inr/` | Future Exp1 training under the new schema |

Unrelated (keep): `step0_inventory/`, `step1_traditional_dti/`.
