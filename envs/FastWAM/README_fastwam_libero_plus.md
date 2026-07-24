# FastWAM LIBERO-Plus evaluation

`run_fastwam_libero_plus.sh` evaluates the 10,030 LIBERO-Plus robustness tasks
with seeds `1`, `7`, and `42`.  One worker is launched per selected GPU and
loads the FastWAM checkpoint once per seed.  The upstream LIBERO-Plus default
of one episode per task is used.

Run on eight GPUs:

```bash
GPUS="0,1,2,3,4,5,6,7" \
  bash /mnt/afs/zhengmingkai/raozf/benchmark/FastWAM/run_fastwam_libero_plus.sh
```

The Python environment defaults to `/root/envs/fastwam_plus/bin/python`.
Results are written below `FastWAM/results_plus/<timestamp>/`:

```text
seed1/{worker_*.json,summary.json,summary.md,logs/,progress/}
seed7/{worker_*.json,summary.json,summary.md,logs/,progress/}
seed42/{worker_*.json,summary.json,summary.md,logs/,progress/}
three_seed_summary.json
summary.md                  # seed 1/7/42 rows plus their arithmetic mean
three_seed_summary.md       # compatibility copy of summary.md
total.log                   # run lifecycle and current seven-dimension rates
```

`three_seed_summary.json` contains the requested arithmetic mean, pooled
success rate, per-suite metrics, all seven LIBERO-Plus robustness columns, and
difficulty-level breakdowns.  Every seed-level `summary.md` reports Camera,
Robot, Language, Light, Background, Noise, Layout, Total, and the underlying
success/episode counts.  The root `summary.md` follows the UnifoLM-VLA table
layout and computes each `Mean` cell from the three unrounded seed rates, so
every random seed has equal weight.  `total.log` receives live `[progress]`
lines as workers finish tasks and `[metrics]` snapshots whenever a seed
finishes.  Both show Camera, Robot, Language, Light, Background, Noise, Layout,
and Total.  By default every completed task produces a live update; set
`PROGRESS_LOG_EVERY=10` (or another positive interval) to reduce log volume.
Monitor a running evaluation with `tail -f <RESULT_ROOT>/total.log`.

Short dry-run:

```bash
DRY_RUN=1 GPUS="0,1" MAX_TASKS=20 \
  bash /mnt/afs/zhengmingkai/raozf/benchmark/FastWAM/run_fastwam_libero_plus.sh
```

Useful overrides include `TASK_SUITES`, `NUM_TRIALS_PER_TASK`, `MAX_TASKS`,
`MAX_STEPS`, `RESULT_BASE`, `RESULT_ROOT`, and `FAIL_FAST=1`.
