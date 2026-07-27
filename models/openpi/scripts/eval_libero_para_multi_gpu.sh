#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: eval_libero_para_multi_gpu.sh [--seed N] [-- EVAL_ARG ...]

Examples:
  GPUS="0 1 2 3" bash scripts/eval_libero_para_multi_gpu.sh --seed 7
  GPUS="0,1" MAX_TASKS=20 MAX_STEPS=5 \
    bash scripts/eval_libero_para_multi_gpu.sh --seed 7 -- --no-save-video

Every GPU loads an independent copy of the same pi05_libero PyTorch
model.safetensors checkpoint. LIBERO-Para tasks are globally truncated by
MAX_TASKS and then split round-robin across workers.

Key environment variables:
  GPUS                    Space/comma-separated physical GPU IDs. By default,
                          use CUDA_VISIBLE_DEVICES or every GPU visible to torch.
  WORLD_SIZE              Number of GPUS entries to use (default: all entries).
                          NUM_WORKERS is accepted as a compatibility alias.
  BASE_PORT               First WebSocket policy-server port (default: 14000).
  SEED                    Global experiment seed (default: 7).
  SERVER_SEED_BASE        Server seed for rank 0 (default: SEED); rank r uses
                          SERVER_SEED_BASE+r.
  TORCH_CHECKPOINT_DIR    Converted OpenPI pi05_libero checkpoint containing
                          model.safetensors and assets/.../norm_stats.json.
  MAX_TASKS               Global task limit before sharding; -1 means all 4092.
  MAX_STEPS               Primitive environment steps per task (default: 300).
  REPLAN_STEPS            Actions executed per OpenPI action chunk (default: 5).
  SAVE_VIDEO              true/false (default: false).
  OUTPUT_DIR              Seed output directory. Defaults under
                          openpi/logs/libero_para/openpi/ (never results/).
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
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$OPENPI_ROOT/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-para}"
EVAL_SCRIPT="$SCRIPT_DIR/eval_libero_para.py"
SERVER_SCRIPT="$SCRIPT_DIR/serve_policy.py"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_libero_para.py"
MERGE_SCRIPT="$SCRIPT_DIR/merge_libero_para_results.py"

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /root/miniconda3/envs/openpi/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/openpi/bin/python
elif [[ -x /root/envs/openpi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_plus/bin/python
elif [[ -x /root/envs/openpi-plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi-plus/bin/python
elif [[ -x /root/envs/openpi_libero/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_libero/bin/python
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  echo "[error] Cannot find an OpenPI Python environment. Set PYTHON=/path/to/python." >&2
  exit 1
fi

TORCH_CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR:-}"  # set to the converted pi05_libero PyTorch checkpoint directory
JAX_CHECKPOINT_DIR="${JAX_CHECKPOINT_DIR:-}"      # set to the official pi05_libero JAX checkpoint directory
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-14000}"
SERVER_SEED_BASE="${SERVER_SEED_BASE:-$SEED}"
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
INITIAL_STATE_INDEX="${INITIAL_STATE_INDEX:-0}"
CLIENT_RETRIES="${CLIENT_RETRIES:-2}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
SERVER_READY_POLL_S="${SERVER_READY_POLL_S:-2}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-2}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

[[ "$SEED" =~ ^-?[0-9]+$ ]] || fail "SEED must be an integer"
[[ "$SERVER_SEED_BASE" =~ ^-?[0-9]+$ ]] || fail "SERVER_SEED_BASE must be an integer"
[[ -x "$PYTHON_BIN" ]] || fail "Python not found: $PYTHON_BIN"
[[ -f "$EVAL_SCRIPT" ]] || fail "Evaluator not found: $EVAL_SCRIPT"
[[ -f "$SERVER_SCRIPT" ]] || fail "Policy server not found: $SERVER_SCRIPT"
[[ -f "$MONITOR_SCRIPT" ]] || fail "Monitor not found: $MONITOR_SCRIPT"
[[ -f "$MERGE_SCRIPT" ]] || fail "Merge script not found: $MERGE_SCRIPT"
if [[ ! -f "$TORCH_CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "[error] Missing OpenPI PyTorch checkpoint: $TORCH_CHECKPOINT_DIR/model.safetensors" >&2
  echo "[error] Convert the JAX checkpoint first:" >&2
  echo "[error]   PYTHONPATH=$OPENPI_ROOT/src $PYTHON_BIN $OPENPI_ROOT/examples/convert_jax_model_to_pytorch.py \\" >&2
  echo "[error]     --checkpoint-dir $JAX_CHECKPOINT_DIR --config-name $POLICY_CONFIG \\" >&2
  echo "[error]     --output-path $TORCH_CHECKPOINT_DIR --precision bfloat16" >&2
  exit 1
fi
NORM_STATS="$TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json"
if [[ ! -f "$NORM_STATS" ]]; then
  echo "[error] Missing LIBERO norm stats: $NORM_STATS" >&2
  echo "[error] Copy them from: $JAX_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" >&2
  exit 1
fi

for argument in "${EXTRA_EVAL_ARGS[@]}"; do
  case "$argument" in
    --num-shards|--num-shards=*|--num_shards|--num_shards=*|\
    --shard-index|--shard-index=*|--shard_index|--shard_index=*|\
    --seed|--seed=*|--server-seed|--server-seed=*|\
    --max-tasks|--max-tasks=*|--output-dir|--output-dir=*|\
    --checkpoint-dir|--checkpoint-dir=*|--policy-config|--policy-config=*|\
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
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_TASKS must be -1 or positive"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || fail "NUM_STEPS_WAIT must be non-negative"
[[ "$REPLAN_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "REPLAN_STEPS must be positive"
[[ "$RESIZE_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "RESIZE_SIZE must be positive"
[[ "$INITIAL_STATE_INDEX" =~ ^[0-9]+$ ]] || fail "INITIAL_STATE_INDEX must be non-negative"
[[ "$CLIENT_RETRIES" =~ ^[0-9]+$ ]] || fail "CLIENT_RETRIES must be non-negative"

LIBERO_INTERNAL_ROOT="$LIBERO_PARA_ROOT/libero/libero"
BDDL_DIR="${BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_para}"
INIT_DIR="${INIT_DIR:-$LIBERO_INTERNAL_ROOT/init_files/libero_para}"
GOAL_BDDL_DIR="${GOAL_BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_goal}"
[[ -d "$BDDL_DIR" && -d "$INIT_DIR" && -d "$GOAL_BDDL_DIR" ]] || fail "LIBERO-Para data is incomplete"
BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
INIT_COUNT="$(find "$INIT_DIR" -maxdepth 1 -type f -name 'eval*.pruned_init' -printf '.' | wc -c)"
GOAL_COUNT="$(find "$GOAL_BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
[[ "$BDDL_COUNT" == "4092" ]] || fail "Expected 4092 LIBERO-Para BDDLs, found $BDDL_COUNT"
[[ "$INIT_COUNT" == "10" && "$GOAL_COUNT" == "10" ]] || fail "Expected 10 init files and goal BDDLs"
if (( MAX_TASKS > 0 && MAX_TASKS < BDDL_COUNT )); then
  EXPECTED_TOTAL="$MAX_TASKS"
else
  EXPECTED_TOTAL="$BDDL_COUNT"
fi
(( WORLD_SIZE <= EXPECTED_TOTAL )) || fail "WORLD_SIZE exceeds the selected task count"

RUN_NAME="${RUN_NAME:-openpi_para_seed${SEED}_${WORLD_SIZE}gpu_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$OPENPI_ROOT/logs/libero_para/openpi}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_ROOT/$RUN_NAME/seed$SEED}"
WORKERS_DIR="$OUTPUT_DIR/workers"
PROCESS_LOG_DIR="$OUTPUT_DIR/logs"
RUNTIME_DIR="$OUTPUT_DIR/runtime"
LIBERO_CONFIG_PATH="$RUNTIME_DIR/libero_config"

if [[ -f "$OUTPUT_DIR/meta.json" || -f "$OUTPUT_DIR/summary.json" || -d "$WORKERS_DIR" ]]; then
  fail "Output directory already contains multi-GPU evaluation data: $OUTPUT_DIR"
fi
mkdir -p "$WORKERS_DIR" "$PROCESS_LOG_DIR/servers" "$PROCESS_LOG_DIR/workers"
mkdir -p "$LIBERO_CONFIG_PATH" "$RUNTIME_DIR/datasets"
{
  printf 'benchmark_root: %s\n' "$LIBERO_INTERNAL_ROOT"
  printf 'bddl_files: %s\n' "$LIBERO_INTERNAL_ROOT/bddl_files"
  printf 'init_states: %s\n' "$LIBERO_INTERNAL_ROOT/init_files"
  printf 'assets: %s\n' "$LIBERO_INTERNAL_ROOT/assets"
  printf 'datasets: %s\n' "$RUNTIME_DIR/datasets"
} > "$LIBERO_CONFIG_PATH/config.yaml"

export PYTHONPATH="$LIBERO_PARA_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
unset DISPLAY XAUTHORITY

CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" - "$TORCH_CHECKPOINT_DIR" <<'PY'
import importlib
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
for module_name in ("torch", "openpi", "openpi_client", "libero.libero.envs", "websockets"):
    importlib.import_module(module_name)

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA/MACA is unavailable in the selected OpenPI environment")
with (checkpoint / "config.json").open("r", encoding="utf-8") as handle:
    config = json.load(handle)
if int(config.get("action_dim", -1)) != 32 or int(config.get("action_horizon", -1)) != 10:
    raise SystemExit(f"Unexpected converted OpenPI checkpoint config: {config}")
print(f"[check] torch={torch.__version__}")
print(f"[check] GPU 0={torch.cuda.get_device_name(0)}")
print(f"[check] OpenPI PyTorch checkpoint={checkpoint}")
PY

echo "[info] OpenPI PyTorch LIBERO-Para multi-GPU evaluation"
echo "[info] Global seed: $SEED"
echo "[info] GPUs: ${GPU_ARRAY[*]:0:$WORLD_SIZE}"
echo "[info] Workers: $WORLD_SIZE"
echo "[info] PyTorch checkpoint on every GPU: $TORCH_CHECKPOINT_DIR"
echo "[info] Tasks: $EXPECTED_TOTAL (round-robin after global truncation)"
echo "[info] Server seeds: $SERVER_SEED_BASE-$((SERVER_SEED_BASE + WORLD_SIZE - 1))"
echo "[info] Server ports: $BASE_PORT-$((BASE_PORT + WORLD_SIZE - 1))"
echo "[info] Output: $OUTPUT_DIR"

server_is_ready() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY' >/dev/null 2>&1
import sys
import websockets.sync.client

try:
    with websockets.sync.client.connect(
        f"ws://{sys.argv[1]}:{sys.argv[2]}",
        open_timeout=1.0,
        close_timeout=1.0,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    ):
        pass
except Exception:
    raise SystemExit(1)
PY
}

assert_port_free() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

cd "$OPENPI_ROOT"
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  server_seed="$((SERVER_SEED_BASE + rank))"
  server_log="$PROCESS_LOG_DIR/servers/rank_${rank}_gpu_${gpu}.log"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" \
    LOCAL_RANK="$rank" \
    WORLD_SIZE="$WORLD_SIZE" \
    "$PYTHON_BIN" -u "$SERVER_SCRIPT" \
      --env LIBERO \
      --server-rank "$rank" \
      --server-world-size "$WORLD_SIZE" \
      --port "$BASE_PORT" \
      --pytorch-device cuda:0 \
      --require-cuda \
      --seed "$server_seed" \
      policy:checkpoint \
      --policy.config "$POLICY_CONFIG" \
      --policy.dir "$TORCH_CHECKPOINT_DIR" \
    >"$server_log" 2>&1 &
  SERVER_PIDS+=("$!")
done

echo "[info] Started $WORLD_SIZE PyTorch policy servers; waiting for model loading..."
deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
while true; do
  ready=0
  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    if server_is_ready "$((BASE_PORT + rank))"; then
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
      tail -n 180 "$server_log" >&2 || true
      exit 1
    fi
  done
  if (( SECONDS >= deadline )); then
    echo "[error] Only $ready/$WORLD_SIZE OpenPI policy servers became ready." >&2
    for server_log in "$PROCESS_LOG_DIR/servers"/*.log; do
      echo "[error] Tail of $server_log" >&2
      tail -n 120 "$server_log" >&2 || true
    done
    exit 1
  fi
  sleep "$SERVER_READY_POLL_S"
done
echo "[info] All $WORLD_SIZE OpenPI PyTorch policy servers are ready."

declare -A PID_TO_RANK=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  port="$((BASE_PORT + rank))"
  server_seed="$((SERVER_SEED_BASE + rank))"
  worker_output="$WORKERS_DIR/rank_${rank}"
  worker_log="$PROCESS_LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$worker_output"

  worker_args=(
    --bddl-dir "$BDDL_DIR"
    --init-dir "$INIT_DIR"
    --goal-bddl-dir "$GOAL_BDDL_DIR"
    --checkpoint-dir "$TORCH_CHECKPOINT_DIR"
    --policy-config "$POLICY_CONFIG"
    --host "$HOST"
    --port "$port"
    --seed "$SEED"
    --server-seed "$server_seed"
    --num-shards "$WORLD_SIZE"
    --shard-index "$rank"
    --output-dir "$worker_output"
    --max-steps "$MAX_STEPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --replan-steps "$REPLAN_STEPS"
    --resize-size "$RESIZE_SIZE"
    --initial-state-index "$INITIAL_STATE_INDEX"
    --client-retries "$CLIENT_RETRIES"
    --server-wait-timeout-s "$SERVER_READY_TIMEOUT_S"
    --server-wait-poll-s "$SERVER_READY_POLL_S"
  )
  [[ "$MAX_TASKS" != "-1" ]] && worker_args+=(--max-tasks "$MAX_TASKS")
  if is_true "$SAVE_VIDEO"; then
    worker_args+=(--save-video)
  else
    worker_args+=(--no-save-video)
  fi
  (( ${#EXTRA_EVAL_ARGS[@]} == 0 )) || worker_args+=("${EXTRA_EVAL_ARGS[@]}")

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" \
    LOCAL_RANK="$rank" \
    WORLD_SIZE="$WORLD_SIZE" \
    "$PYTHON_BIN" -u "$EVAL_SCRIPT" "${worker_args[@]}" \
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
      tail -n 180 "$PROCESS_LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log" >&2 || true
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

echo "[ok] OpenPI PyTorch multi-GPU seed $SEED evaluation complete."
echo "[ok] Summary: $OUTPUT_DIR/summary.json"
echo "[ok] Process logs: $PROCESS_LOG_DIR"
