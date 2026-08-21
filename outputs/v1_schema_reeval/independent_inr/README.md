# v1 schema re-eval — Independent INR (no retrain)

These numbers use the **new metrics schema** (MAE / RMSE / Pearson r / DWI RelMSE).

Weights are **copied/scored from v0 training**, not retrained:

`outputs/v0_preschema/independent_inr/`

Look at:

- `summary.csv`
- `aggregate.csv` / `aggregate.md`
- `<sid>/{best.pt, maps.npz, metrics.json}`

If you later **train** after the schema change, that run goes to:

`outputs/v1_schema_train/independent_inr/`
