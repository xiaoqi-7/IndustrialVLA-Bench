# FastWAM LIBERO-Para evaluation

`run_fastwam_libero_para.sh` evaluates the 4,092 paraphrased LIBERO-Para BDDL
tasks with seeds `1`, `7`, and `42`.  Each selected GPU loads FastWAM once and
evaluates a round-robin task shard.  By default each paraphrase is run for one
episode, matching the upstream LIBERO-Para evaluation scripts.

Run on all eight GPUs:

```bash
GPUS="0,1,2,3,4,5,6,7" \
  bash /mnt/afs/zhengmingkai/raozf/benchmark/FastWAM/run_fastwam_libero_para.sh
```

Short validation run (still performs all three seeds, but only two tasks per
seed):

```bash
GPUS="0,1" MAX_TASKS=2 NUM_TRIALS_PER_TASK=1 MAX_STEPS=5 \
  NUM_STEPS_WAIT=0 NUM_INFERENCE_STEPS=1 REPLAN_STEPS=2 \
  bash /mnt/afs/zhengmingkai/raozf/benchmark/FastWAM/run_fastwam_libero_para.sh
```

Use `DRY_RUN=1` to scan and shard tasks without loading the model.  Paths can
be overridden with `MODEL_PATH`, `LIBERO_PARA_ROOT`, `PYTHON_BIN`,
`FASTWAM_SOURCE_ROOT`, and `LEROBOT_ZIP`.

Results are written as:

```text
<RESULT_ROOT>/
  seed1/{worker_*.json,summary.json,logs/}
  seed7/{worker_*.json,summary.json,logs/}
  seed42/{worker_*.json,summary.json,logs/}
  three_seed_summary.json
```

The requested arithmetic mean is `mean_success_rate`; the summary also
contains the pooled rate and breakdowns by original eval scenario and
paraphrase type.
