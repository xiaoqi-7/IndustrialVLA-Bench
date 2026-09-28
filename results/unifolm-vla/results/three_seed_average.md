# UnifoLM-VLA three-seed averages

以下数值均为三颗独立运行种子的等权平均，单位为成功率。分种子明细保留在各 benchmark 子目录的 `three_seed_average.md` 中。

## LIBERO

| LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | Overall |
|---:|---:|---:|---:|---:|
| **95.27%** | **97.73%** | **100.00%** | **98.67%** | **97.92%** |

每个子套件每颗种子包含 500 episodes。Overall 为四个子套件的等权平均；三种子总计 5,875/6,000，精确 Overall 为 97.9167%。

## LIBERO-plus

| Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **57.03%** | **68.37%** | **91.08%** | **93.70%** | **95.14%** | **79.25%** | **79.19%** | **79.18%** |


## LIBERO-Para

| Eval | Original task | Three-seed average |
|---:|---|---:|
| eval0 | Open the middle layer of the drawer | 98.08% |
| eval1 | Open the top layer of the drawer and put the bowl inside | 47.40% |
| eval2 | Push the plate to the front of the stove | 65.68% |
| eval3 | Put the bowl on the plate | 85.06% |
| eval4 | Put the bowl on the stove | 84.12% |
| eval5 | Put the bowl on the top of the drawer | 84.96% |
| eval6 | Put the cream cheese on the bowl | 80.86% |
| eval7 | Put the wine bottle on the rack | 93.79% |
| eval8 | Put the wine bottle on the top of the drawer | 87.71% |
| eval9 | Turn on the stove | 94.20% |
| **Overall** | All 4,092 paraphrases per run | **82.24%** |

实际记录种子为 1、8、44（源目录名为 `seed_1`、`seed_7`、`seed_42`）。总计 10,096/12,276；精确 Overall 为 82.2418%。
