#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar

# Evaluate OpenPI on LIBERO-plus with the local PyTorch checkpoint.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$OPENPI_ROOT/.." && pwd)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$OPENPI_ROOT/LIBERO-plus}"
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
  echo "[error] Cannot find openpi-plus python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-${TASK_SUITE_NAME:- libero_10  libero_object}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
GPUS="${GPUS:-${RANKS:-0 5 6 7}}"
PORT="${PORT:-6666}"
HOST="${HOST:-127.0.0.1}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-3600}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
RUN_NAME="${RUN_NAME:-torch_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$OPENPI_ROOT/logs/libero_plus_eval/$RUN_NAME}"
SERVER_LOG_ROOT="${SERVER_LOG_ROOT:-$OPENPI_ROOT/logs/openpi_plus_servers_torch}"
VIDEO_OUT_ROOT="${VIDEO_OUT_ROOT:-$OPENPI_ROOT/data/libero_plus/videos/$RUN_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$OPENPI_ROOT/logs/libero_plus_eval/$RUN_NAME/results}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$OPENPI_ROOT/data/libero_plus/config}"

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

SAVE_VIDEO_ARGS=()
case "${SAVE_VIDEO,,}" in
  false|0|no|n) SAVE_VIDEO_ARGS=(--no-save-video) ;;
esac

export PYTHONPATH="$LIBERO_PLUS_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset DISPLAY
unset XAUTHORITY

LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"
mkdir -p "$LOG_ROOT" "$SERVER_LOG_ROOT" "$VIDEO_OUT_ROOT" "$RESULT_ROOT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

cd "$OPENPI_ROOT"

if [[ ! -f "$TORCH_CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "[error] Missing PyTorch checkpoint: $TORCH_CHECKPOINT_DIR/model.safetensors" >&2
  echo "[error] Convert the successful JAX checkpoint first:" >&2
  echo "[error]   $PYTHON_BIN examples/convert_jax_model_to_pytorch.py --checkpoint-dir /mnt/afs/raozf/models/pi05_libero/pi05_libero --config-name pi05_libero --output-path $TORCH_CHECKPOINT_DIR --precision bfloat16" >&2
  exit 1
fi
if [[ ! -f "$TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "[error] Missing LIBERO norm stats: $TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" >&2
  exit 1
fi

echo "[info] OpenPI root: $OPENPI_ROOT"
echo "[info] LIBERO-plus root: $LIBERO_PLUS_ROOT"
echo "[info] Python: $PYTHON_BIN"
echo "[info] Torch checkpoint: $TORCH_CHECKPOINT_DIR"
echo "[info] Task suites: $TASK_SUITES"
echo "[info] GPUs: $GPUS, world size: $WORLD_SIZE, base port: $PORT"
echo "[info] Logs: $LOG_ROOT"

SERVER_PIDS=()
cleanup() {
  if (( ${#SERVER_PIDS[@]} > 0 )); then
    kill "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Start one policy server per local rank. CUDA_VISIBLE_DEVICES maps it to a physical GPU.
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

# Evaluate all LIBERO-plus perturbation tasks from the four original LIBERO suites.
for task_suite in $TASK_SUITES; do
  suite_log_root="$LOG_ROOT/$task_suite"
  suite_video_root="$VIDEO_OUT_ROOT/$task_suite"
  suite_result_root="$RESULT_ROOT/$task_suite"
  mkdir -p "$suite_log_root" "$suite_video_root" "$suite_result_root"

  echo "[info] Starting LIBERO-plus suite: $task_suite"
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
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "[info] Finished suite: $task_suite"
done

RESULT_FILES=("$RESULT_ROOT"/**/*.jsonl)
if (( ${#RESULT_FILES[@]} == 0 )); then
  echo "[error] No result JSONL files found under $RESULT_ROOT." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_libero_plus.py "${RESULT_FILES[@]}" --model "$MODEL_NAME" \
  | tee "$LOG_ROOT/summary.md"

echo "[info] Results: $RESULT_ROOT"
echo "[info] Summary: $LOG_ROOT/summary.md"
