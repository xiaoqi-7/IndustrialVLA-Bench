# UnifoLM-VLA LIBERO-Para three-seed average

目录名与 `meta.json` 中记录的实际随机种子不完全一致，所以下表同时列出两者。每颗种子均包含完整的 4,092 episodes。

| Source directory | Recorded seed | Successes | Episodes | Success rate |
|---|---:|---:|---:|---:|
| `seed_1` | 1 | 3351 | 4092 | 81.89% |
| `seed_7` | 8 | 3393 | 4092 | 82.92% |
| `seed_42` | 44 | 3352 | 4092 | 81.92% |
| **Mean** | — | — | — | **82.24%** |

| Eval split | Three-seed mean |
|---:|---:|
| eval0 | 98.08% |
| eval1 | 47.40% |
| eval2 | 65.68% |
| eval3 | 85.06% |
| eval4 | 84.12% |
| eval5 | 84.96% |
| eval6 | 80.86% |
| eval7 | 93.79% |
| eval8 | 87.71% |
| eval9 | 94.20% |

精确等权均值为 82.2418%，population standard deviation 为 0.4782 个百分点。合计 10,096/12,276，因每颗种子规模相同，pooled success rate 也为 82.2418%。
