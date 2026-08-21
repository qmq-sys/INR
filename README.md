# INR — Shared INR + Subject Latent for DTI/DKI

## 输出分三个版本（不要混用）

详见 `outputs/OUTPUTS.md`。

| 目录 | 含义 |
|------|------|
| `outputs/v0_preschema/independent_inr/` | 改指标**前**的 29 人训练（旧文件结构） |
| `outputs/v1_schema_reeval/independent_inr/` | 同一套 v0 权重，用**新指标**重评分（没有重训） |
| `outputs/v1_schema_train/independent_inr/` | 改指标后的**新训练**（目前空；跑 step3 才写入） |

看新指标总表：`outputs/v1_schema_reeval/independent_inr/summary.csv`

---

## 固定评价指标

1. 参数 vs WLS：FA/MD/AD/RD 的 MAE / RMSE / Pearson r  
2. DWI：MAE + RelMSE  
3. 训练：final_loss / best_loss / best_epoch / training_time_sec  

每人默认：`best.pt` + `maps.npz` + `metrics.json`

---

## 运行

```powershell
cd E:\BaiduNetdiskDownload\INR
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 新训练会写入 v1_schema_train/，不会覆盖 v0 或 re-eval
python scripts/step3_independent_inr.py --skip-done --skip-traditional-if-exists

# 只用旧权重按新指标重评（写入 v1_schema_reeval/）
python scripts/reformat_independent_to_schema.py
```
