# OpenPI-Torch three-seed averages

单位均为成功率。LIBERO 按现有有效种子统计并排除缺失结果；LIBERO-plus 和 LIBERO-Para 使用三颗完整种子的等权平均。分种子明细保留在各 benchmark 子目录的 `three_seed_average.md` 中。

## LIBERO

| LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | Overall |
|---:|---:|---:|---:|---:|
| **93.13%** | **97.60%** | **97.89%** | **98.33%** | **96.74%** |

按现有结果统计：每个子套件先计算各颗有效种子的成功率，再对有数据的种子等权平均；完全缺失的种子或 episodes 不记录、不补零。中断日志在中断前已经完成且有成功/失败记录的 episodes 会计入。有效覆盖为 LIBERO-10 `928/1500`（2 颗种子）、Goal `1300/1500`（3 颗种子）、Object `934/1500`（3 颗种子）、Spatial `1500/1500`（3 颗种子）。Overall 是四个分项均值的等权平均，精确值为 96.7385%。

## LIBERO-plus

| Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **70.63%** | **75.12%** | **85.97%** | **96.76%** | **95.82%** | **87.05%** | **86.47%** | **84.38%** |

各项由每颗种子的成功数/任务数计算，再对种子 1、7、42 等权平均。总计 25,390/30,090；精确 Overall 为 84.3802%。

## LIBERO-Para

| Eval | Original task | Three-seed average |
|---:|---|---:|
| eval0 | Open the middle layer of the drawer | 78.61% |
| eval1 | Open the top layer of the drawer and put the bowl inside | 71.54% |
| eval2 | Push the plate to the front of the stove | 36.78% |
| eval3 | Put the bowl on the plate | 71.68% |
| eval4 | Put the bowl on the stove | 79.32% |
| eval5 | Put the bowl on the top of the drawer | 75.04% |
| eval6 | Put the cream cheese on the bowl | 82.97% |
| eval7 | Put the wine bottle on the rack | 48.35% |
| eval8 | Put the wine bottle on the top of the drawer | 80.54% |
| eval9 | Turn on the stove | 87.84% |
| **Overall** | All 4,092 paraphrases per seed | **71.33%** |

种子为 1、7、42。总计 8,756/12,276；精确 Overall 为 71.3262%。
