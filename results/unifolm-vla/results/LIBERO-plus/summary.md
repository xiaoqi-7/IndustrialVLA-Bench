# UnifoLM-VLA LIBERO-plus three-seed summary

| Seed | Camera | Robot | Language | Light | Background | Noise | Layout | Total | Evaluated tasks |
|-----:|-------:|------:|---------:|------:|-----------:|------:|-------:|------:|----------------:|
| 1 | 56.7 | 67.2 | 90.3 | 93.5 | 95.2 | 79.5 | 78.6 | 78.8 | 10030 |
| 7* | 57.4 | 68.5 | 92.2 | 94.0 | 95.4 | 79.7 | 79.9 | 79.7 | 5015 |
| 42 | 56.9 | 69.4 | 90.8 | 93.5 | 94.9 | 78.6 | 79.1 | 79.1 | 10030 |
| **Mean** | **57.0** | **68.4** | **91.1** | **93.7** | **95.1** | **79.3** | **79.2** | **79.2** | — |

Counts:
Seed 1: 7899/10030 (2131 failures)
Seed 7: 3996/5015 (1019 failures; incomplete)
Seed 42: 7934/10030 (2096 failures)

The mean is the arithmetic mean of the three unrounded per-seed success rates, giving each seed equal weight.

\* Seed 7 is incomplete: only 5,015 of 10,030 tasks were evaluated because ranks 1, 2, 4, 7, 8, 11, 13, and 15 failed with CUDA out-of-memory errors. The 5,015 missing tasks are excluded rather than counted as failures.
