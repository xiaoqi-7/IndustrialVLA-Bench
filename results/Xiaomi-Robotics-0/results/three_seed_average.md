# Xiaomi-Robotics-0 three-seed averages

以下数值均为三颗独立运行种子的等权平均，单位为成功率。分种子明细保留在各 benchmark 子目录的 `three_seed_average.md` 中。

## LIBERO

| LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | Overall |
|---:|---:|---:|---:|---:|
| **95.80%** | **97.93%** | **100.00%** | **98.67%** | **98.10%** |

每个子套件每颗种子包含 500 episodes。Overall 为四个子套件的等权平均；三种子总计 5,886/6,000，精确 Overall 为 98.10%。`seed_42/libero_object` 的 100.00% 从 10 个任务日志各 50/50 的最终记录复算。

## LIBERO-plus

| Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **40.21%** | **55.63%** | **89.03%** | **94.51%** | **90.43%** | **86.78%** | **75.87%** | **74.50%** |

各项由每颗种子的成功数/任务数计算，再对种子 1、7、42 等权平均。总计 22,417/30,090；精确 Overall 为 74.4998%。

## LIBERO-Para

| Eval | Original task | Three-seed average |
|---:|---|---:|
| eval0 | Open the middle layer of the drawer | 99.60% |
| eval1 | Open the top layer of the drawer and put the bowl inside | 57.40% |
| eval2 | Push the plate to the front of the stove | 54.02% |
| eval3 | Put the bowl on the plate | 96.29% |
| eval4 | Put the bowl on the stove | 81.80% |
| eval5 | Put the bowl on the top of the drawer | 73.50% |
| eval6 | Put the cream cheese on the bowl | 75.67% |
| eval7 | Put the wine bottle on the rack | 48.67% |
| eval8 | Put the wine bottle on the top of the drawer | 82.11% |
| eval9 | Turn on the stove | 88.81% |
| **Overall** | All 4,092 paraphrases per run | **75.72%** |

实际记录种子为 42、7、44（源目录名为 `seed_1`、`seed_7`、`seed_42`）。总计 9,296/12,276；精确 Overall 为 75.7250%。
