#!/usr/bin/env bash
set -euo pipefail

# Evaluate OpenPI on LIBERO-plus. Defaults mirror eval.sh while making the
# LIBERO-plus installation and config explicit.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$OPENPI_ROOT/.." && pwd)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$BENCHMARK_ROOT/LIBERO-plus}"

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /root/envs/openpi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_plus/bin/python
elif [[ -x /root/envs/openpi-plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi-plus/bin/python
else
  echo "[error] Cannot find openpi-plus python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
GPUS="${GPUS:-${RANKS:-3 4 5 6 7}}"
PORT="${PORT:-8015}"
HOST="${HOST:-127.0.0.1}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-3600}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
LOG_ROOT="${LOG_ROOT:-$OPENPI_ROOT/logs/libero_plus_eval/$TASK_SUITE_NAME}"
SERVER_LOG_ROOT="${SERVER_LOG_ROOT:-$OPENPI_ROOT/logs/openpi_plus_servers}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-$OPENPI_ROOT/data/libero_plus/videos/$TASK_SUITE_NAME}"
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
mkdir -p "$LOG_ROOT" "$SERVER_LOG_ROOT" "$VIDEO_OUT_PATH" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

cd "$OPENPI_ROOT"

echo "[info] OpenPI root: $OPENPI_ROOT"
echo "[info] LIBERO-plus root: $LIBERO_PLUS_ROOT"
echo "[info] Python: $PYTHON_BIN"
echo "[info] Task suite: $TASK_SUITE_NAME"
echo "[info] GPUs: $GPUS, world size: $WORLD_SIZE, base port: $PORT"
echo "[info] Logs: $LOG_ROOT"

# Start evaluators first; each local rank waits for its matching policy server.
for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
  "$PYTHON_BIN" examples/libero/main.py \
    --eval-rank "$rank" \
    --eval-world-size "$WORLD_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --task-suite-name "$TASK_SUITE_NAME" \
    --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
    --video-out-path "$VIDEO_OUT_PATH" \
    "${SAVE_VIDEO_ARGS[@]}" \
    --server-wait-timeout-s "$SERVER_WAIT_TIMEOUT_S" \
    > "$LOG_ROOT/eval_rank_${rank}.log" 2>&1 &
done

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
    > "$SERVER_LOG_ROOT/server_rank_${rank}.log" 2>&1 &
done

echo "[info] Started LIBERO-plus evaluation and policy servers."
echo "[info] Eval logs: tail -f $LOG_ROOT/eval_rank_<rank>.log"
echo "[info] Server logs: tail -f $SERVER_LOG_ROOT/server_rank_<rank>.log"
