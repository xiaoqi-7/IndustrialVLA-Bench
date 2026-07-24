# Xiaomi-Robotics-0 LIBERO three-seed average

单位均为成功率；每个子套件每颗种子包含 500 episodes，`Overall` 为四个子套件的等权平均。

| Seed | LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | Overall |
|---:|---:|---:|---:|---:|---:|
| 1 | 94.60% | 96.60% | 100.00% | 98.80% | 97.50% |
| 7 | 96.00% | 98.60% | 100.00% | 98.60% | 98.30% |
| 42 | 96.80% | 98.60% | 100.00% | 98.60% | 98.50% |
| **Mean** | **95.80%** | **97.93%** | **100.00%** | **98.67%** | **98.10%** |

三种子合计为 5,886/6,000 成功；由于各颗种子的 episode 数相同，pooled success rate 与等权三种子均值相同，均为 98.10%。`seed_42/libero_object` 缺少 `merged_results.json`，其 100.00% 来自 10 个任务日志中各 50/50 成功的最终记录。
