#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar

# Evaluate OpenPI on the standard LIBERO benchmark with the local PyTorch checkpoint.
# This intentionally mirrors eval_plus.sh, but points at ./LIBERO instead of ./LIBERO-plus
# and summarizes standard LIBERO suites instead of LIBERO-plus perturbation columns.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LIBERO_ROOT="${LIBERO_ROOT:-$OPENPI_ROOT/LIBERO}"
TORCH_CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR:-/mnt/afs/raozf/models/pi05_libero/pi05_libero_pytorch}"
MODEL_NAME="${MODEL_NAME:-OpenPI-Torch}"

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x /root/miniconda3/envs/openpi/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/openpi/bin/python
elif [[ -x /root/envs/openpi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_plus/bin/python
elif [[ -x /root/envs/openpi-plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi-plus/bin/python
else
  echo "[error] Cannot find openpi python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

# Full standard LIBERO evaluation: two 10-task suites, 50 rollouts per task.
TASK_SUITES="${TASK_SUITES:-libero_object libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
GPUS="${GPUS:-${RANKS:-0 1 2 5 7}}"
GPUS="${GPUS//,/ }"
PORT="${PORT:-6666}"
HOST="${HOST:-127.0.0.1}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-3600}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
RUN_NAME="${RUN_NAME:-torch_libero_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$OPENPI_ROOT/logs/libero_eval/$RUN_NAME}"
SERVER_LOG_ROOT="${SERVER_LOG_ROOT:-$OPENPI_ROOT/logs/openpi_libero_servers_torch/$RUN_NAME}"
VIDEO_OUT_ROOT="${VIDEO_OUT_ROOT:-$OPENPI_ROOT/data/libero/videos/$RUN_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$OPENPI_ROOT/logs/libero_eval/$RUN_NAME/results}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$OPENPI_ROOT/data/libero/config}"
LIBERO_DATASETS_PATH="${LIBERO_DATASETS_PATH:-$OPENPI_ROOT/data/libero/datasets}"

read -r -a GPU_ARRAY <<< "$GPUS"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ARRAY[@]}}"
if (( WORLD_SIZE < 1 )); then
  echo "[error] WORLD_SIZE must be >= 1." >&2
  exit 1
fi
if (( ${#GPU_ARRAY[@]} < WORLD_SIZE )); then
  echo "[error] GPUS='$GPUS' has fewer entries than WORLD_SIZE=$WORLD_SIZE." >&2
  exit 1
fi

if [[ ! -d "$LIBERO_ROOT" ]]; then
  echo "[error] Missing LIBERO root: $LIBERO_ROOT" >&2
  exit 1
fi

LIBERO_INTERNAL_ROOT="$LIBERO_ROOT/libero/libero"
for required_dir in \
  "$LIBERO_INTERNAL_ROOT/bddl_files" \
  "$LIBERO_INTERNAL_ROOT/init_files" \
  "$LIBERO_INTERNAL_ROOT/assets"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[error] Missing LIBERO directory: $required_dir" >&2
    exit 1
  fi
done

if [[ ! -f "$TORCH_CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "[error] Missing PyTorch checkpoint: $TORCH_CHECKPOINT_DIR/model.safetensors" >&2
  echo "[error] Convert the JAX checkpoint first, for example:" >&2
  echo "[error]   $PYTHON_BIN examples/convert_jax_model_to_pytorch.py --checkpoint-dir /mnt/afs/raozf/models/pi05_libero/pi05_libero --config-name pi05_libero --output-path $TORCH_CHECKPOINT_DIR --precision bfloat16" >&2
  exit 1
fi
if [[ ! -f "$TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "[error] Missing LIBERO norm stats: $TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" >&2
  exit 1
fi

SAVE_VIDEO_ARGS=()
case "${SAVE_VIDEO,,}" in
  false|0|no|n) SAVE_VIDEO_ARGS=(--no-save-video) ;;
esac

export PYTHONPATH="$LIBERO_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
# LIBERO stores initial states as legacy torch pickles. PyTorch 2.6 changed
# torch.load's default to weights_only=True, which cannot read these files.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset DISPLAY
unset XAUTHORITY

mkdir -p "$LOG_ROOT" "$SERVER_LOG_ROOT" "$VIDEO_OUT_ROOT" "$RESULT_ROOT" "$LIBERO_CONFIG_PATH" "$LIBERO_DATASETS_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_DATASETS_PATH
YAML

cd "$OPENPI_ROOT"

echo "[info] OpenPI root: $OPENPI_ROOT"
echo "[info] LIBERO root: $LIBERO_ROOT"
echo "[info] Python: $PYTHON_BIN"
echo "[info] Torch checkpoint: $TORCH_CHECKPOINT_DIR"
echo "[info] Task suites: $TASK_SUITES"
echo "[info] Trials per task: $NUM_TRIALS_PER_TASK"
echo "[info] GPUs: $GPUS, world size: $WORLD_SIZE, base port: $PORT"
echo "[info] Logs: $LOG_ROOT"

SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
  if (( ${#EVAL_PIDS[@]} > 0 )); then
    kill "${EVAL_PIDS[@]}" >/dev/null 2>&1 || true
    wait "${EVAL_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  if (( ${#SERVER_PIDS[@]} > 0 )); then
    kill "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
    wait "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Start one policy server per local rank. CUDA_VISIBLE_DEVICES maps each process
# to one physical GPU, so the server uses cuda:0 inside that process.
for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$rank]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
  "$PYTHON_BIN" scripts/serve_policy.py \
    --env LIBERO \
    --server-rank "$rank" \
    --server-world-size "$WORLD_SIZE" \
    --port "$PORT" \
    --pytorch-device cuda:0 \
    --require-cuda \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$TORCH_CHECKPOINT_DIR" \
    > "$SERVER_LOG_ROOT/server_rank_${rank}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[info] Started PyTorch policy servers."
echo "[info] Server logs: tail -f $SERVER_LOG_ROOT/server_rank_<rank>.log"

for task_suite in $TASK_SUITES; do
  suite_log_root="$LOG_ROOT/$task_suite"
  suite_video_root="$VIDEO_OUT_ROOT/$task_suite"
  suite_result_root="$RESULT_ROOT/$task_suite"
  mkdir -p "$suite_log_root" "$suite_video_root" "$suite_result_root"

  echo "[info] Starting LIBERO suite: $task_suite"
  pids=()
  for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
    "$PYTHON_BIN" examples/libero/main.py \
      --eval-rank "$rank" \
      --eval-world-size "$WORLD_SIZE" \
      --host "$HOST" \
      --port "$PORT" \
      --task-suite-name "$task_suite" \
      --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
      --video-out-path "$suite_video_root" \
      --result-out-path "$suite_result_root/rank_${rank}.jsonl" \
      "${SAVE_VIDEO_ARGS[@]}" \
      --server-wait-timeout-s "$SERVER_WAIT_TIMEOUT_S" \
      > "$suite_log_root/eval_rank_${rank}.log" 2>&1 &
    pids+=("$!")
    EVAL_PIDS+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  EVAL_PIDS=()
  echo "[info] Finished suite: $task_suite"
done

RESULT_FILES=("$RESULT_ROOT"/**/*.jsonl)
if (( ${#RESULT_FILES[@]} == 0 )); then
  echo "[error] No result JSONL files found under $RESULT_ROOT." >&2
  exit 1
fi

"$PYTHON_BIN" - "$MODEL_NAME" "$NUM_TRIALS_PER_TASK" "$TASK_SUITES" "${RESULT_FILES[@]}" <<'PY' | tee "$LOG_ROOT/summary.md"
import json
import pathlib
import sys
from collections import defaultdict

model_name = sys.argv[1]
trials_per_task = int(sys.argv[2])
requested_suites = sys.argv[3].split()
result_paths = [pathlib.Path(path) for path in sys.argv[4:]]

suite_labels = {
    "libero_spatial": "Libero Spatial",
    "libero_object": "Libero Object",
    "libero_goal": "Libero Goal",
    "libero_10": "Libero 10",
    "libero_90": "Libero 90",
}
suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

stats = defaultdict(lambda: [0, 0])
task_episode_counts = defaultdict(lambda: defaultdict(int))
for path in sorted(result_paths):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            suite = record.get("task_suite", "unknown")
            task_id = int(record.get("task_id", -1))
            stats[suite][0] += int(bool(record.get("success")))
            stats[suite][1] += 1
            task_episode_counts[suite][task_id] += 1

# Fail instead of reporting a partial score if a rank exited early or a task
# did not run the requested number of episodes.
expected_task_counts = {"libero_object": 10, "libero_10": 10}
for suite in requested_suites:
    task_count = expected_task_counts.get(suite)
    if task_count is None:
        continue
    expected_per_task = trials_per_task
    actual_task_counts = task_episode_counts[suite]
    bad_tasks = {
        task_id: count
        for task_id, count in sorted(actual_task_counts.items())
        if task_id not in range(task_count) or count != expected_per_task
    }
    missing_tasks = {
        task_id: 0
        for task_id in range(task_count)
        if task_id not in actual_task_counts
    }
    bad_tasks.update(missing_tasks)
    if bad_tasks:
        raise SystemExit(
            f"Incomplete {suite} evaluation: expected {task_count} tasks x "
            f"{expected_per_task} episodes; task counts={dict(sorted(bad_tasks.items()))}"
        )

observed_suites = [suite for suite in suite_order if stats[suite][1]]
observed_suites.extend(sorted(suite for suite in stats if suite not in suite_order and stats[suite][1]))

def rate(suite: str) -> float:
    successes, episodes = stats[suite]
    return 100.0 * successes / episodes if episodes else 0.0

header = ["Model", *[suite_labels.get(suite, suite) for suite in observed_suites], "Average"]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")

rates = [rate(suite) for suite in observed_suites]
average = sum(rates) / len(rates) if rates else 0.0
print("| " + " | ".join([model_name, *[f"{value:.1f}" for value in rates], f"{average:.1f}"]) + " |")

print("\nCounts:")
total_successes = 0
total_episodes = 0
for suite in observed_suites:
    successes, episodes = stats[suite]
    total_successes += successes
    total_episodes += episodes
    print(f"{suite_labels.get(suite, suite)}: {successes}/{episodes}")
print(f"Total: {total_successes}/{total_episodes}")
PY

echo "[info] Results: $RESULT_ROOT"
echo "[info] Summary: $LOG_ROOT/summary.md"
