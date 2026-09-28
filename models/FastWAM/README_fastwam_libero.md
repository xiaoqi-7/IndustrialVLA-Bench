# FastWAM LIBERO evaluation

`run_fastwam_libero.sh` evaluates the supplied checkpoint with three seeds
(`1`, `7`, `42`) and writes the arithmetic mean and the episode-weighted pooled
success rate to `three_seed_summary.json`.  It also writes the compact
four-subtask report `four_task_summary.json`.  Every selected GPU runs one worker;
the worker loads the policy once and receives a round-robin shard of the LIBERO
tasks.

The supplied snapshot is a LeRobot checkpoint (`model.safetensors` plus local
Wan/UMT5 sidecars).  The `lerobot.zip` in this benchmark is an older LeRobot
snapshot without the `fastwam` policy module, so the evaluator uses the local
reference implementation in `FastWAM/src` and streams the LeRobot weights into
it.  This avoids downloading the 5B backbone or any sidecar at evaluation time.

Run the default four LIBERO suites on all visible GPUs:

```bash
bash FastWAM/run_fastwam_libero.sh
```

Useful overrides:

```bash
GPUS="0,1,2,3" NUM_TRIALS_PER_TASK=50 \
  RESULT_ROOT=/tmp/fastwam-libero \
  bash FastWAM/run_fastwam_libero.sh

# A short smoke test (one task and one episode):
GPUS=0 MAX_TASKS=1 NUM_TRIALS_PER_TASK=1 MAX_STEPS=5 \
  NUM_STEPS_WAIT=0 NUM_INFERENCE_STEPS=1 REPLAN_STEPS=2 \
  bash FastWAM/run_fastwam_libero.sh
```

The default paths are the paths in the request.  They can be changed with
`PYTHON_BIN`, `MODEL_PATH`, `LIBERO_ROOT`, `FASTWAM_SOURCE_ROOT`, and
`LEROBOT_ZIP`.  The evaluator writes a run-local LIBERO config (override its
location with `FASTWAM_LIBERO_CONFIG_PATH`).  Set `MUJOCO_GL`/`PYOPENGL_PLATFORM`
when the host needs a different headless renderer (the default is `osmesa`).

Output layout:

```text
<RESULT_ROOT>/
  seed1/{worker_*.json,summary.json,logs/}
  seed7/{worker_*.json,summary.json,logs/}
  seed42/{worker_*.json,summary.json,logs/}
  three_seed_summary.json       # full aggregate (including per-seed details)
  four_task_summary.json        # rates for the four LIBERO suites
  four_task_summary.md          # same rates as a compact table
```

If an interrupted run has partial worker logs, the standalone resume utility
can continue only the missing episodes and append to the existing worker log:

```bash
/root/envs/fastwam_libero/bin/python FastWAM/resume_libero_workers.py --dry-run
/root/envs/fastwam_libero/bin/python FastWAM/resume_libero_workers.py
```

The four subtasks are `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`.  Each entry in `four_task_summary.json` includes per-seed rates,
the arithmetic mean, and the pooled successes/episodes and success rate.
