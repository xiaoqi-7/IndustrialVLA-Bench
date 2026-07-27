#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: eval_libero_para_multi_gpu.sh [--seed N] [-- EVAL_ARG ...]

Examples:
  GPUS="0 1 2 3" bash scripts/eval_libero_para_multi_gpu.sh --seed 7
  GPUS="0,1" MAX_TASKS=20 MAX_STEPS=5 \
    bash scripts/eval_libero_para_multi_gpu.sh --seed 7 -- --no-log-trajectories

Each GPU loads an independent copy of the same libero_goal checkpoint. The
selected LIBERO-Para tasks are globally truncated by MAX_TASKS and then split
round-robin across workers. This does not load the four LIBERO-Plus models.

Key environment variables:
  GPUS                    Space/comma-separated physical GPU IDs. By default,
                          use CUDA_VISIBLE_DEVICES or every GPU visible to torch.
  WORLD_SIZE              Number of GPUS entries to use (default: all entries).
                          NUM_WORKERS is accepted as a compatibility alias.
  BASE_PORT               First model-server port (default: 13000).
  SEED                    Global experiment seed (default: 7).
  SERVER_SEED_BASE        Server seed for rank 0 (default: SEED); rank r uses
                          SERVER_SEED_BASE+r.
  MAX_TASKS               Global task limit before sharding; -1 means all 4092.
  MAX_STEPS               Primitive environment steps per task (default: 300).
  N_ACTION_STEPS          GR00T actions executed per inference chunk (default: 8).
  OUTPUT_DIR              Seed output directory. Defaults under
                          Isaac-GR00T/logs/libero_para/gr00t/ (never results/).
  MONITOR_INTERVAL_S      Global success-rate refresh interval (default: 2).

Merged files are written to OUTPUT_DIR/{eval0..9,meta,progress,summary}.json.
Worker data stays in OUTPUT_DIR/workers and process logs in OUTPUT_DIR/logs.
EOF
}

SEED="${SEED:-1}"  # paper protocol: three runs with SEED=1, 7, 42
while (( $# > 0 )); do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { echo "[error] --seed requires an integer" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown launcher argument: $1 (put evaluator args after --)" >&2
      exit 2
      ;;
  esac
done
EXTRA_EVAL_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$REPO_DIR/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-para}"
SINGLE_LAUNCHER="$SCRIPT_DIR/eval_libero_para.sh"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_libero_para.py"
MERGE_SCRIPT="$SCRIPT_DIR/merge_libero_para_results.py"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_libero}"
PYTHON_BIN="$CONDA_ENV/bin/python"

MODEL_PATH="${MODEL_PATH:-}"  # set to the GR00T-N1.7 libero_goal checkpoint directory
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-13000}"
SERVER_SEED_BASE="${SERVER_SEED_BASE:-$SEED}"
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
INITIAL_STATE_INDEX="${INITIAL_STATE_INDEX:-0}"
CLIENT_TIMEOUT_MS="${CLIENT_TIMEOUT_MS:-120000}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-true}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-2}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ "$SEED" =~ ^-?[0-9]+$ ]] || fail "SEED must be an integer"
[[ "$SERVER_SEED_BASE" =~ ^-?[0-9]+$ ]] || fail "SERVER_SEED_BASE must be an integer"
[[ -x "$PYTHON_BIN" ]] || fail "Python not found: $PYTHON_BIN"
[[ -x "$SINGLE_LAUNCHER" ]] || fail "Launcher not executable: $SINGLE_LAUNCHER"
[[ -f "$MONITOR_SCRIPT" ]] || fail "Monitor not found: $MONITOR_SCRIPT"
[[ -f "$MERGE_SCRIPT" ]] || fail "Merge script not found: $MERGE_SCRIPT"
[[ -f "$MODEL_PATH/config.json" ]] || fail "libero_goal checkpoint not found: $MODEL_PATH"

for argument in "${EXTRA_EVAL_ARGS[@]}"; do
  case "$argument" in
    --num-shards|--num-shards=*|--num_shards|--num_shards=*|\
    --shard-index|--shard-index=*|--shard_index|--shard_index=*|\
    --seed|--seed=*|--server-seed|--server-seed=*|\
    --max-tasks|--max-tasks=*|--output-dir|--output-dir=*|\
    --host|--host=*|--port|--port=*)
      fail "The multi-GPU launcher controls evaluator argument: $argument"
      ;;
  esac
done

if [[ -n "${GPUS:-}" ]]; then
  GPU_TEXT="$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_TEXT="$CUDA_VISIBLE_DEVICES"
else
  GPU_COUNT="$($PYTHON_BIN - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
  [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "No CUDA GPU is visible to torch"
  GPU_TEXT=""
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
    GPU_TEXT+="${GPU_TEXT:+ }$gpu_index"
  done
fi
GPU_TEXT="${GPU_TEXT//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_TEXT"
(( ${#GPU_ARRAY[@]} > 0 )) || fail "GPUS must contain at least one GPU ID"

if [[ -n "${WORLD_SIZE:-}" && -n "${NUM_WORKERS:-}" && "$WORLD_SIZE" != "$NUM_WORKERS" ]]; then
  fail "WORLD_SIZE and NUM_WORKERS disagree"
fi
WORLD_SIZE="${WORLD_SIZE:-${NUM_WORKERS:-${#GPU_ARRAY[@]}}}"
[[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "WORLD_SIZE must be a positive integer"
(( WORLD_SIZE <= ${#GPU_ARRAY[@]} )) || fail "WORLD_SIZE exceeds the number of GPUS entries"

declare -A SEEN_GPUS=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU ID must be a non-negative integer: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || fail "Duplicate GPU ID: $gpu"
  SEEN_GPUS[$gpu]=1
done

[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || fail "BASE_PORT must be a positive integer"
(( BASE_PORT + WORLD_SIZE - 1 <= 65535 )) || fail "Server port range exceeds 65535"
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || {
  fail "MAX_TASKS must be -1 or positive"
}
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "$N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "N_ACTION_STEPS must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || fail "NUM_STEPS_WAIT must be non-negative"

BDDL_DIR="${BDDL_DIR:-$LIBERO_PARA_ROOT/libero/libero/bddl_files/libero_para}"
[[ -d "$BDDL_DIR" ]] || fail "LIBERO-Para BDDL directory not found: $BDDL_DIR"
BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
[[ "$BDDL_COUNT" == "4092" ]] || fail "Expected 4092 LIBERO-Para BDDLs, found $BDDL_COUNT"
if (( MAX_TASKS > 0 && MAX_TASKS < BDDL_COUNT )); then
  EXPECTED_TOTAL="$MAX_TASKS"
else
  EXPECTED_TOTAL="$BDDL_COUNT"
fi
(( WORLD_SIZE <= EXPECTED_TOTAL )) || fail "WORLD_SIZE exceeds the selected task count"

RUN_NAME="${RUN_NAME:-gr00t_libero_para_seed${SEED}_${WORLD_SIZE}gpu_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/libero_para/gr00t}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_ROOT/$RUN_NAME/seed$SEED}"
WORKERS_DIR="$OUTPUT_DIR/workers"
PROCESS_LOG_DIR="$OUTPUT_DIR/logs"
SERVER_RUNTIME_DIR="$OUTPUT_DIR/server_runtime"

if [[ -f "$OUTPUT_DIR/meta.json" || -f "$OUTPUT_DIR/summary.json" || -d "$WORKERS_DIR" ]]; then
  fail "Output directory already contains multi-GPU evaluation data: $OUTPUT_DIR"
fi
mkdir -p "$WORKERS_DIR" "$PROCESS_LOG_DIR/servers" "$PROCESS_LOG_DIR/workers"
mkdir -p "$SERVER_RUNTIME_DIR"

echo "[info] Isaac-GR00T LIBERO-Para multi-GPU evaluation"
echo "[info] Global seed: $SEED"
echo "[info] GPUs: ${GPU_ARRAY[*]:0:$WORLD_SIZE}"
echo "[info] Workers: $WORLD_SIZE"
echo "[info] Checkpoint on every GPU: $MODEL_PATH (libero_goal only)"
echo "[info] Tasks: $EXPECTED_TOTAL (round-robin after global truncation)"
echo "[info] Server seeds: $SERVER_SEED_BASE-$((SERVER_SEED_BASE + WORLD_SIZE - 1))"
echo "[info] Server ports: $BASE_PORT-$((BASE_PORT + WORLD_SIZE - 1))"
echo "[info] Output: $OUTPUT_DIR"

port_is_listening() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5).close()
except OSError:
    raise SystemExit(1)
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
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill -INT -- "-$pid" >/dev/null 2>&1 || kill -INT "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  for pid in "${EVAL_PIDS[@]}"; do
    stop_group "$pid"
  done
  for pid in "${SERVER_PIDS[@]}"; do
    stop_group "$pid"
  done
  sleep 2
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  assert_port_free "$((BASE_PORT + rank))"
done

for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  port="$((BASE_PORT + rank))"
  server_seed="$((SERVER_SEED_BASE + rank))"
  server_output="$SERVER_RUNTIME_DIR/rank_${rank}"
  server_log="$PROCESS_LOG_DIR/servers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$server_output"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CONDA_ENV="$CONDA_ENV" \
    MODEL_PATH="$MODEL_PATH" \
    LIBERO_PARA_ROOT="$LIBERO_PARA_ROOT" \
    HOST="$HOST" \
    PORT="$port" \
    SEED="$SEED" \
    SERVER_SEED="$server_seed" \
    NUM_SHARDS=1 \
    SHARD_INDEX=0 \
    OUTPUT_DIR="$server_output" \
    RUNTIME_DIR="$server_output/runtime" \
    SERVER_READY_TIMEOUT_S="$SERVER_READY_TIMEOUT_S" \
    bash "$SINGLE_LAUNCHER" server --seed "$SEED" --server-seed "$server_seed" \
    >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[info] Started $WORLD_SIZE model servers; waiting for checkpoint loading..."
deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
while true; do
  ready=0
  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    if port_is_listening "$((BASE_PORT + rank))"; then
      ready=$((ready + 1))
    fi
  done
  (( ready == WORLD_SIZE )) && break

  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    pid="${SERVER_PIDS[$rank]}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      gpu="${GPU_ARRAY[$rank]}"
      server_log="$PROCESS_LOG_DIR/servers/rank_${rank}_gpu_${gpu}.log"
      echo "[error] Server rank $rank on GPU $gpu exited during startup." >&2
      tail -n 160 "$server_log" >&2 || true
      exit 1
    fi
  done
  if (( SECONDS >= deadline )); then
    echo "[error] Only $ready/$WORLD_SIZE model servers became ready." >&2
    for server_log in "$PROCESS_LOG_DIR/servers"/*.log; do
      echo "[error] Tail of $server_log" >&2
      tail -n 100 "$server_log" >&2 || true
    done
    exit 1
  fi
  sleep 2
done
echo "[info] All $WORLD_SIZE model servers are ready."

declare -A PID_TO_RANK=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  port="$((BASE_PORT + rank))"
  server_seed="$((SERVER_SEED_BASE + rank))"
  worker_output="$WORKERS_DIR/rank_${rank}"
  worker_log="$PROCESS_LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$worker_output"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CONDA_ENV="$CONDA_ENV" \
    MODEL_PATH="$MODEL_PATH" \
    LIBERO_PARA_ROOT="$LIBERO_PARA_ROOT" \
    HOST="$HOST" \
    PORT="$port" \
    SEED="$SEED" \
    SERVER_SEED="$server_seed" \
    NUM_SHARDS="$WORLD_SIZE" \
    SHARD_INDEX="$rank" \
    OUTPUT_DIR="$worker_output" \
    RUNTIME_DIR="$worker_output/runtime" \
    MAX_TASKS="$MAX_TASKS" \
    MAX_STEPS="$MAX_STEPS" \
    NUM_STEPS_WAIT="$NUM_STEPS_WAIT" \
    N_ACTION_STEPS="$N_ACTION_STEPS" \
    INITIAL_STATE_INDEX="$INITIAL_STATE_INDEX" \
    CLIENT_TIMEOUT_MS="$CLIENT_TIMEOUT_MS" \
    SAVE_VIDEO="$SAVE_VIDEO" \
    LOG_TRAJECTORIES="$LOG_TRAJECTORIES" \
    SERVER_READY_TIMEOUT_S="$SERVER_READY_TIMEOUT_S" \
    bash "$SINGLE_LAUNCHER" eval --seed "$SEED" --server-seed "$server_seed" \
      -- "${EXTRA_EVAL_ARGS[@]}" \
    >"$worker_log" 2>&1 &
  pid="$!"
  EVAL_PIDS+=("$pid")
  PID_TO_RANK["$pid"]="$rank"
done

echo "[info] Started $WORLD_SIZE evaluation shards; printing global real-time accuracy."
"$PYTHON_BIN" -u "$MONITOR_SCRIPT" \
  --workers-dir "$WORKERS_DIR" \
  --total "$EXPECTED_TOTAL" \
  --num-workers "$WORLD_SIZE" \
  --seed "$SEED" \
  --interval "$MONITOR_INTERVAL_S" &
MONITOR_PID="$!"

ACTIVE_PIDS=("${EVAL_PIDS[@]}")
worker_status=0
while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  finished_pid=""
  set +e
  wait -n -p finished_pid "${ACTIVE_PIDS[@]}"
  exit_code="$?"
  set -e
  rank="${PID_TO_RANK[$finished_pid]:-unknown}"

  next_active=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    [[ "$pid" == "$finished_pid" ]] || next_active+=("$pid")
  done
  ACTIVE_PIDS=("${next_active[@]}")

  if (( exit_code != 0 )); then
    echo "[error] Evaluation rank $rank failed with exit code $exit_code." >&2
    if [[ "$rank" != "unknown" ]]; then
      gpu="${GPU_ARRAY[$rank]}"
      tail -n 160 "$PROCESS_LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log" >&2 || true
    fi
    worker_status="$exit_code"
    break
  fi
  echo "[info] Evaluation rank $rank finished successfully."
done

if (( worker_status != 0 )); then
  EVAL_PIDS=("${ACTIVE_PIDS[@]}")
  exit "$worker_status"
fi
EVAL_PIDS=()

if [[ -n "$MONITOR_PID" ]]; then
  wait "$MONITOR_PID" || true
  MONITOR_PID=""
fi

"$PYTHON_BIN" "$MERGE_SCRIPT" \
  --workers-dir "$WORKERS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --expected-total "$EXPECTED_TOTAL" \
  --num-shards "$WORLD_SIZE" \
  --seed "$SEED"

echo "[ok] Multi-GPU seed $SEED evaluation complete."
echo "[ok] Summary: $OUTPUT_DIR/summary.json"
echo "[ok] Process logs: $PROCESS_LOG_DIR"
