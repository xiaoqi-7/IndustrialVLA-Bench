# Isaac GR00T three-seed averages

以下数值均为三颗独立运行种子的等权平均，单位为成功率。分种子明细保留在各 benchmark 子目录的 `three_seed_average.md` 中。

## LIBERO

| LIBERO-10 | LIBERO-Goal | LIBERO-Object | LIBERO-Spatial | Overall |
|---:|---:|---:|---:|---:|
| N/A | N/A | N/A | N/A | **N/A** |

## LIBERO-plus

| Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **64.40%** | **38.40%** | **83.88%** | **94.87%** | **93.72%** | **84.36%** | **74.94%** | **74.43%** |

七个扰动项先在每颗种子内对四个 LIBERO 子套件做等权宏平均，再对种子 1、7、42 等权平均。Overall 来自各颗种子的总成功率；总计 22,395/30,090。

## LIBERO-Para

| Eval | Original task | Three-seed average |
|---:|---|---:|
| eval0 | Open the middle layer of the drawer | 93.83% |
| eval1 | Open the top layer of the drawer and put the bowl inside | 43.58% |
| eval2 | Push the plate to the front of the stove | 84.32% |
| eval3 | Put the bowl on the plate | 75.04% |
| eval4 | Put the bowl on the stove | 96.20% |
| eval5 | Put the bowl on the top of the drawer | 77.64% |
| eval6 | Put the cream cheese on the bowl | 22.06% |
| eval7 | Put the wine bottle on the rack | 78.45% |
| eval8 | Put the wine bottle on the top of the drawer | 84.00% |
| eval9 | Turn on the stove | 87.36% |
| **Overall** | All 4,092 paraphrases per run | **74.26%** |

实际记录种子为 1、0、8（源目录名为 `seed1`、`seed7`、`seed42`）。总计 9,116/12,276；精确 Overall 为 74.2587%。
