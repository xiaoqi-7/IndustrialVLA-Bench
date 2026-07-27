#!/usr/bin/env bash
set -Eeuo pipefail

# Parallel Xiaomi-Robotics-0 evaluation for LIBERO-Para.
# One model server and one deterministic task shard are assigned to each GPU.

usage() {
  cat <<'EOF'
Usage: eval_libero_para_multi_gpu.sh [-- EVAL_ARG ...]

Example:
  GPUS="0 1 2 3 4 5 6 7" \
    bash scripts/eval_libero_para_multi_gpu.sh

Important environment variables:
  GPUS                    Space/comma-separated physical GPU IDs (default: CUDA_VISIBLE_DEVICES or 0)
  NUM_WORKERS             Number of GPUs/workers to use (default: number of GPUS entries)
  BASE_PORT               First model-server port (default: 11000)
  RUN_NAME                Run directory name (timestamped default)
  OUTPUT_DIR              Root result directory
  MAX_TASKS               Global task limit before sharding; -1 runs all 4092
  MAX_STEPS               Maximum steps per task (default: 300)
  SEED                    Evaluation seed (default: 7)
  SAVE_VIDEO              true/false (default: false)
  MONITOR_INTERVAL_S      Global success-rate refresh interval (default: 2)

The merged output uses the official layout at OUTPUT_DIR/{eval0..9,meta,summary}.json.
Per-worker data and logs are retained under OUTPUT_DIR/workers and OUTPUT_DIR/logs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_EVAL_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XIAOMI_ROOT="${XIAOMI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$XIAOMI_ROOT/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-para}"
SINGLE_LAUNCHER="$SCRIPT_DIR/eval_libero_para.sh"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_libero_para.py"
MERGE_SCRIPT="$SCRIPT_DIR/merge_libero_para_results.py"
CONDA_ENV="${CONDA_ENV:-/root/envs/xiaomi_libero}"
PYTHON_BIN="$CONDA_ENV/bin/python"

MODEL_PATH="${MODEL_PATH:-}"  # set to the Xiaomi-Robotics-0-LIBERO checkpoint directory
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-12000}"
SEED="${SEED:-1}"  # paper protocol: three runs with SEED=1, 7, 42
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-2}"

GPU_TEXT="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_TEXT="${GPU_TEXT//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_TEXT"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ARRAY[@]}}"

RUN_NAME="${RUN_NAME:-xiaomi_robotics_0_seed${SEED}_${NUM_WORKERS}gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$XIAOMI_ROOT/results/libero_para/$RUN_NAME}"
WORKERS_DIR="$OUTPUT_DIR/workers"
LOG_DIR="$OUTPUT_DIR/logs"
SERVER_RUNTIME_DIR="$OUTPUT_DIR/server_runtime"

BDDL_DIR="${BDDL_DIR:-$LIBERO_PARA_ROOT/libero/libero/bddl_files/libero_para}"
BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -x "$PYTHON_BIN" ]] || fail "Python not found: $PYTHON_BIN"
[[ -x "$SINGLE_LAUNCHER" ]] || fail "Launcher not executable: $SINGLE_LAUNCHER"
[[ -f "$MONITOR_SCRIPT" ]] || fail "Monitor not found: $MONITOR_SCRIPT"
[[ -f "$MERGE_SCRIPT" ]] || fail "Merge script not found: $MERGE_SCRIPT"
[[ -f "$MODEL_PATH/config.json" ]] || fail "Checkpoint not found: $MODEL_PATH"
(( ${#GPU_ARRAY[@]} > 0 )) || fail "GPUS must contain at least one GPU ID"
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "NUM_WORKERS must be a positive integer"
(( NUM_WORKERS <= ${#GPU_ARRAY[@]} )) || fail "NUM_WORKERS exceeds the number of GPUS entries"
[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || fail "BASE_PORT must be a positive integer"
(( BASE_PORT + NUM_WORKERS - 1 <= 65535 )) || fail "Server port range exceeds 65535"
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_TASKS must be -1 or positive"

if (( MAX_TASKS > 0 && MAX_TASKS < BDDL_COUNT )); then
  EXPECTED_TOTAL="$MAX_TASKS"
else
  EXPECTED_TOTAL="$BDDL_COUNT"
fi
(( NUM_WORKERS <= EXPECTED_TOTAL )) || fail "More workers than selected tasks"

declare -A SEEN_GPUS=()
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU ID must be a non-negative integer: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || fail "Duplicate GPU ID: $gpu"
  SEEN_GPUS[$gpu]=1
done

if [[ -f "$OUTPUT_DIR/meta.json" || -f "$OUTPUT_DIR/summary.json" || -d "$WORKERS_DIR" ]]; then
  fail "Output directory already contains parallel evaluation data: $OUTPUT_DIR"
fi
mkdir -p "$WORKERS_DIR" "$LOG_DIR/servers" "$LOG_DIR/workers" "$SERVER_RUNTIME_DIR"

echo "[info] Multi-GPU LIBERO-Para evaluation"
echo "[info] GPUs: ${GPU_ARRAY[*]:0:$NUM_WORKERS}"
echo "[info] Workers: $NUM_WORKERS"
echo "[info] Tasks: $EXPECTED_TOTAL (round-robin sharding)"
echo "[info] Server ports: $BASE_PORT-$((BASE_PORT + NUM_WORKERS - 1))"
echo "[info] Output: $OUTPUT_DIR"

port_is_listening() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.4)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

assert_port_free() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((sys.argv[1], int(sys.argv[2])))
except OSError as exc:
    raise SystemExit(f"Port {sys.argv[1]}:{sys.argv[2]} is unavailable: {exc}")
finally:
    sock.close()
PY
}

SERVER_PIDS=()
EVAL_PIDS=()
MONITOR_PID=""

stop_group() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -INT -- "-$pid" >/dev/null 2>&1 || kill -INT "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  for pid in "${EVAL_PIDS[@]}"; do
    stop_group "$pid"
  done
  for pid in "${SERVER_PIDS[@]}"; do
    stop_group "$pid"
  done
  sleep 1
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  assert_port_free "$((BASE_PORT + rank))"
done

for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  port="$((BASE_PORT + rank))"
  server_output="$SERVER_RUNTIME_DIR/rank_${rank}"
  server_log="$LOG_DIR/servers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$server_output"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CONDA_ENV="$CONDA_ENV" \
    MODEL_PATH="$MODEL_PATH" \
    LIBERO_PARA_ROOT="$LIBERO_PARA_ROOT" \
    HOST="$HOST" \
    PORT="$port" \
    OUTPUT_DIR="$server_output" \
    RUNTIME_DIR="$server_output/runtime" \
    SERVER_READY_TIMEOUT_S="$SERVER_READY_TIMEOUT_S" \
    bash "$SINGLE_LAUNCHER" server \
    >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[info] Started $NUM_WORKERS GPU model servers; waiting for model loading..."
deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
while true; do
  ready=0
  for ((rank = 0; rank < NUM_WORKERS; rank++)); do
    if port_is_listening "$((BASE_PORT + rank))"; then
      ready=$((ready + 1))
    fi
  done
  if (( ready == NUM_WORKERS )); then
    break
  fi

  for ((rank = 0; rank < NUM_WORKERS; rank++)); do
    pid="${SERVER_PIDS[$rank]}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      gpu="${GPU_ARRAY[$rank]}"
      echo "[error] Server rank $rank on GPU $gpu exited during startup." >&2
      tail -n 120 "$LOG_DIR/servers/rank_${rank}_gpu_${gpu}.log" >&2 || true
      exit 1
    fi
  done
  if (( SECONDS >= deadline )); then
    echo "[error] Only $ready/$NUM_WORKERS model servers became ready." >&2
    for log in "$LOG_DIR/servers"/*.log; do
      echo "===== $log =====" >&2
      tail -n 80 "$log" >&2 || true
    done
    exit 1
  fi
  sleep 2
done
echo "[info] All $NUM_WORKERS GPU model servers are ready."

for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  port="$((BASE_PORT + rank))"
  worker_output="$WORKERS_DIR/rank_${rank}"
  worker_log="$LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$worker_output"

  worker_args=(--num_shards "$NUM_WORKERS" --shard_index "$rank")
  if (( ${#EXTRA_EVAL_ARGS[@]} > 0 )); then
    worker_args+=("${EXTRA_EVAL_ARGS[@]}")
  fi

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CONDA_ENV="$CONDA_ENV" \
    MODEL_PATH="$MODEL_PATH" \
    LIBERO_PARA_ROOT="$LIBERO_PARA_ROOT" \
    HOST="$HOST" \
    PORT="$port" \
    SEED="$SEED" \
    OUTPUT_DIR="$worker_output" \
    RUNTIME_DIR="$worker_output/runtime" \
    MAX_TASKS="$MAX_TASKS" \
    MAX_STEPS="$MAX_STEPS" \
    NUM_STEPS_WAIT="$NUM_STEPS_WAIT" \
    REPLAN_STEPS="$REPLAN_STEPS" \
    SAVE_VIDEO="$SAVE_VIDEO" \
    SERVER_READY_TIMEOUT_S="$SERVER_READY_TIMEOUT_S" \
    bash "$SINGLE_LAUNCHER" eval -- "${worker_args[@]}" \
    >"$worker_log" 2>&1 &
  EVAL_PIDS+=("$!")
done

echo "[info] Started $NUM_WORKERS evaluation shards."
"$PYTHON_BIN" -u "$MONITOR_SCRIPT" \
  --workers-dir "$WORKERS_DIR" \
  --total "$EXPECTED_TOTAL" \
  --interval "$MONITOR_INTERVAL_S" &
MONITOR_PID="$!"

status=0
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  if ! wait "${EVAL_PIDS[$rank]}"; then
    echo "[error] Evaluation rank $rank failed; see $LOG_DIR/workers" >&2
    status=1
  fi
done
EVAL_PIDS=()

if (( status != 0 )); then
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
  for log in "$LOG_DIR/workers"/*.log; do
    echo "===== $log =====" >&2
    tail -n 80 "$log" >&2 || true
  done
  exit 1
fi

if [[ -n "$MONITOR_PID" ]]; then
  wait "$MONITOR_PID" || true
  MONITOR_PID=""
fi

"$PYTHON_BIN" "$MERGE_SCRIPT" \
  --workers-dir "$WORKERS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --expected-total "$EXPECTED_TOTAL" \
  --num-shards "$NUM_WORKERS"

echo "[ok] Multi-GPU evaluation complete."
echo "[ok] Summary: $OUTPUT_DIR/summary.json"
echo "[ok] Worker logs: $LOG_DIR/workers"
