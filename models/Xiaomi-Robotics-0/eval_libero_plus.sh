#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

XIAOMI_ROOT="${XIAOMI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$XIAOMI_ROOT/.." && pwd)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$BENCHMARK_ROOT/LIBERO-plus}"

MODEL_PATH="${1:-${MODEL_PATH:-/mnt/afs/zhengmingkai/raozf/models/Xiaomi-Robotics-0-LIBERO}}"
NUM_PORTS="${2:-${NUM_PORTS:-8}}"
NUM_GPUS="${3:-${NUM_GPUS:-8}}"
LOG_BASE="${4:-${LOG_BASE:-$XIAOMI_ROOT/logs}}"

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /root/envs/xiaomi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/xiaomi_plus/bin/python
else
  echo "[error] Cannot find /root/envs/xiaomi_plus/bin/python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

if ! [[ "$NUM_PORTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NUM_PORTS must be a positive integer." >&2
  exit 1
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NUM_GPUS must be a positive integer." >&2
  exit 1
fi

if [[ ! -f "$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json" ]]; then
  echo "[error] LIBERO-plus task_classification.json not found under $LIBERO_PLUS_ROOT." >&2
  echo "[error] Set LIBERO_PLUS_ROOT=/path/to/LIBERO-plus if needed." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ["lazy_loader"] if importlib.util.find_spec(name) is None]
if missing:
    print(f"[error] Missing Python package(s): {', '.join(missing)}", file=sys.stderr)
    print("[error] Install in /root/envs/xiaomi_plus, e.g. pip install lazy_loader", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as exc:
    print(f"[error] Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("[error] CUDA is not available; Xiaomi-Robotics-0 LIBERO-plus eval must run on GPU.", file=sys.stderr)
    sys.exit(1)
PY
then
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
BASE_PORT="${BASE_PORT:-9999}"
HOST="${HOST:-127.0.0.1}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Xiaomi-Robotics-0}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
SERVER_START_DELAY_S="${SERVER_START_DELAY_S:-5}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-3600}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
WORKER_PROGRESS="${WORKER_PROGRESS:-false}"
GPUS="${GPUS:-$(seq -s ' ' 0 $((NUM_GPUS - 1)))}"

read -r -a GPU_ARRAY <<< "$GPUS"
WORLD_SIZE="${WORLD_SIZE:-$NUM_PORTS}"
if (( WORLD_SIZE < 1 )); then
  echo "[error] WORLD_SIZE must be >= 1." >&2
  exit 1
fi
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPUS must contain at least one GPU id." >&2
  exit 1
fi

LOG_ROOT="$LOG_BASE/libero_plus_eval/$RUN_NAME"
SERVER_LOG_ROOT="$LOG_ROOT/servers"
EVAL_LOG_ROOT="$LOG_ROOT/eval"
RESULT_ROOT="$LOG_ROOT/results"
VIDEO_ROOT="$LOG_ROOT/videos"
SUMMARY_PATH="$LOG_ROOT/summary.md"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LOG_ROOT/libero_config}"
CLASSIFICATION_PATH="$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"
LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"

mkdir -p "$SERVER_LOG_ROOT" "$EVAL_LOG_ROOT" "$RESULT_ROOT" "$VIDEO_ROOT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

export PYTHONPATH="$LIBERO_PLUS_ROOT:$XIAOMI_ROOT:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$LOG_ROOT/hf_modules_cache}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset DISPLAY
unset XAUTHORITY
if [[ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

rm -rf "$HF_MODULES_CACHE"
mkdir -p "$HF_MODULES_CACHE"

SAVE_VIDEO_ARGS=(--no-save-video)
case "${SAVE_VIDEO,,}" in
  true|1|yes|y) SAVE_VIDEO_ARGS=(--save-video) ;;
esac

WORKER_PROGRESS_ARGS=(--no-worker-progress)
case "${WORKER_PROGRESS,,}" in
  true|1|yes|y) WORKER_PROGRESS_ARGS=(--worker-progress) ;;
esac

cd "$XIAOMI_ROOT"

echo "[info] Xiaomi root: $XIAOMI_ROOT"
echo "[info] LIBERO-plus root: $LIBERO_PLUS_ROOT"
echo "[info] Python: $PYTHON_BIN"
echo "[info] Model: $MODEL_PATH"
echo "[info] Task suites: $TASK_SUITES"
echo "[info] GPUs: $GPUS, world size: $WORLD_SIZE, base port: $BASE_PORT"
echo "[info] Logs: $LOG_ROOT"

SERVER_PIDS=()
EVAL_PIDS=()
MONITOR_PID=""
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  if (( ${#EVAL_PIDS[@]} > 0 )); then
    kill "${EVAL_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  if (( ${#SERVER_PIDS[@]} > 0 )); then
    for server_pid in "${SERVER_PIDS[@]}"; do
      kill -- "-$server_pid" >/dev/null 2>&1 || kill "$server_pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT INT TERM

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  port=$((BASE_PORT + rank))
  if ! "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    print(f"[error] Port {host}:{port} is already in use: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
PY
  then
    echo "[error] Stop existing servers or set BASE_PORT to a free port range." >&2
    exit 1
  fi
done

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$((rank % ${#GPU_ARRAY[@]}))]}"
  port=$((BASE_PORT + rank))
  rank_hf_modules_cache="$HF_MODULES_CACHE/rank_${rank}"
  mkdir -p "$rank_hf_modules_cache"
  CUDA_VISIBLE_DEVICES="$gpu" HF_MODULES_CACHE="$rank_hf_modules_cache" setsid "$PYTHON_BIN" deploy/server.py \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$port" \
    > "$SERVER_LOG_ROOT/server_rank_${rank}_gpu_${gpu}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[info] Started $WORLD_SIZE model servers on GPU(s): $GPUS"
echo "[info] Server logs: $SERVER_LOG_ROOT/server_rank_<rank>_gpu_<gpu>.log"
sleep "$SERVER_START_DELAY_S"

echo "[info] Waiting for model servers to listen..."
deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
while true; do
  ready=0
  for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    port=$((BASE_PORT + rank))
    if "$PYTHON_BIN" - "$HOST" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.3)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      ready=$((ready + 1))
    fi
  done

  failed=0
  for pid in "${SERVER_PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      failed=1
    fi
  done

  if (( ready == WORLD_SIZE )); then
    echo "[info] All model servers are ready."
    break
  fi
  if (( failed != 0 || SECONDS >= deadline )); then
    echo "[error] Only $ready/$WORLD_SIZE model servers became ready." >&2
    echo "[error] Server logs:" >&2
    for log_file in "$SERVER_LOG_ROOT"/*.log; do
      echo "===== $log_file =====" >&2
      tail -n 80 "$log_file" >&2 || true
    done
    exit 1
  fi
  sleep 2
done

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$((rank % ${#GPU_ARRAY[@]}))]}"
  port=$((BASE_PORT + rank))
  CUDA_VISIBLE_DEVICES="$gpu" RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
  "$PYTHON_BIN" eval_libero/eval_libero_plus.py \
    --libero-plus-root "$LIBERO_PLUS_ROOT" \
    --classification-path "$CLASSIFICATION_PATH" \
    --task-suites $TASK_SUITES \
    --eval-rank "$rank" \
    --eval-world-size "$WORLD_SIZE" \
    --host "$HOST" \
    --port "$port" \
    --server-wait-timeout-s "$SERVER_WAIT_TIMEOUT_S" \
    --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
    --video-out-path "$VIDEO_ROOT/rank_${rank}" \
    --limit-tasks "$LIMIT_TASKS" \
    --result-jsonl "$RESULT_ROOT/rank_${rank}.jsonl" \
    "${SAVE_VIDEO_ARGS[@]}" \
    "${WORKER_PROGRESS_ARGS[@]}" \
    > "$EVAL_LOG_ROOT/eval_rank_${rank}_gpu_${gpu}.log" 2>&1 &
  EVAL_PIDS+=("$!")
done

echo "[info] Started $WORLD_SIZE eval workers. Eval logs: $EVAL_LOG_ROOT"

case "${SHOW_PROGRESS,,}" in
  false|0|no|n) ;;
  *)
    "$PYTHON_BIN" eval_libero/monitor_libero_plus_progress.py \
      --result-dir "$RESULT_ROOT" \
      --classification-path "$CLASSIFICATION_PATH" \
      --task-suites $TASK_SUITES \
      --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
      --limit-tasks "$LIMIT_TASKS" &
    MONITOR_PID="$!"
    ;;
esac

status=0
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
EVAL_PIDS=()

if [[ -n "$MONITOR_PID" ]]; then
  if (( status == 0 )); then
    wait "$MONITOR_PID" || true
  else
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  MONITOR_PID=""
fi

if (( status != 0 )); then
  echo "[error] Some eval workers failed. Last log lines:" >&2
  for log_file in "$EVAL_LOG_ROOT"/*.log; do
    echo "===== $log_file =====" >&2
    tail -n 80 "$log_file" >&2 || true
  done
  exit 1
fi

RESULT_FILES=("$RESULT_ROOT"/rank_*.jsonl)
if (( ${#RESULT_FILES[@]} == 0 )); then
  echo "[error] No result JSONL files found under $RESULT_ROOT." >&2
  exit 1
fi

"$PYTHON_BIN" eval_libero/summarize_libero_plus.py "${RESULT_FILES[@]}" --model "$MODEL_NAME" | tee "$SUMMARY_PATH"

echo "[info] Results: $RESULT_ROOT"
echo "[info] Summary: $SUMMARY_PATH"
