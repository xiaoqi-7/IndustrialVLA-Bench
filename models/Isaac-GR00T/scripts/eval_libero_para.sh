#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: eval_libero_para.sh [all|smoke|server|eval|check] [--seed N]
                           [--server-seed N] [-- EVAL_ARG ...]

This launcher always uses the single GR00T libero_goal checkpoint because
LIBERO-Para paraphrases the 10 libero_goal tasks.

Modes:
  all      Start one GPU model server, evaluate, then stop the server (default).
  smoke    Same as all, limited to one paraphrase unless MAX_TASKS is set.
  server   Start only the seeded GR00T model server in the foreground.
  eval     Evaluate against an already-running server with the same seed.
  check    Validate paths, data, imports, and GPU availability.

Key environment variables:
  SEED                    Random seed (default: 7; --seed takes precedence)
  SERVER_SEED             Model-server seed (default: SEED)
  CONDA_ENV               Default: /root/envs/gr00t_libero
  MODEL_PATH              Default: .../Gr00t-N1.7-libero/libero_goal
  CUDA_VISIBLE_DEVICES    One GPU ID (default: 0)
  HOST / PORT             Default: 127.0.0.1 / 5555
  MAX_STEPS               Primitive environment steps per task (default: 300)
  N_ACTION_STEPS          Actions executed per GR00T chunk (default: 8)
  MAX_TASKS               -1 for all 4092 tasks (default: -1)
  NUM_SHARDS              Number of deterministic task shards (default: 1)
  SHARD_INDEX             Zero-based shard index (default: 0)
  INITIAL_STATE_INDEX     Fixed LIBERO-Para init-state index (default: 0)
  SAVE_VIDEO              true/false (default: false)
  LOG_TRAJECTORIES        true/false (default: true)
  OUTPUT_DIR              Seed-specific log/result directory

Arguments after -- are appended to scripts/eval_libero_para.py.
EOF
}

MODE="all"
if [[ "${1:-}" =~ ^(all|smoke|server|eval|check)$ ]]; then
  MODE="$1"
  shift
elif [[ "${1:-}" =~ ^(-h|--help|help)$ ]]; then
  usage
  exit 0
fi

SEED="${SEED:-7}"
SERVER_SEED="${SERVER_SEED:-}"
while (( $# > 0 )); do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { echo "[error] --seed requires an integer" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --server-seed)
      [[ $# -ge 2 ]] || { echo "[error] --server-seed requires an integer" >&2; exit 2; }
      SERVER_SEED="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
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
SERVER_SEED="${SERVER_SEED:-$SEED}"

[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "[error] SEED must be an integer" >&2; exit 2; }
[[ "$SERVER_SEED" =~ ^-?[0-9]+$ ]] || {
  echo "[error] SERVER_SEED must be an integer" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$REPO_DIR/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$BENCHMARK_ROOT/LIBERO-Para}"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_libero}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/zhengmingkai/raozf/models/Gr00t-N1.7-libero/libero_goal}"
COSMOS_PATH="${COSMOS_PATH:-/mnt/afs/zhengmingkai/raozf/models/Cosmos-Reason2-2B}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
INITIAL_STATE_INDEX="${INITIAL_STATE_INDEX:-0}"
CLIENT_TIMEOUT_MS="${CLIENT_TIMEOUT_MS:-120000}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
LOG_TRAJECTORIES="${LOG_TRAJECTORIES:-true}"
OFFLINE="${OFFLINE:-1}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"

if [[ "$MODE" == "smoke" ]]; then
  MAX_TASKS="${MAX_TASKS_OVERRIDE:-${MAX_TASKS/-1/1}}"
fi

RUN_NAME="${RUN_NAME:-gr00t_libero_para_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/libero_para/gr00t}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOG_ROOT/$RUN_NAME/seed$SEED}"
RUNTIME_DIR="${RUNTIME_DIR:-$OUTPUT_DIR/runtime}"
SERVER_LOG="${SERVER_LOG:-$OUTPUT_DIR/server.log}"
EVAL_LOG="${EVAL_LOG:-$OUTPUT_DIR/eval.log}"

EVAL_SCRIPT="$SCRIPT_DIR/eval_libero_para.py"
SERVER_SCRIPT="$REPO_DIR/gr00t/eval/run_gr00t_server.py"
LIBERO_INTERNAL_ROOT="$LIBERO_PARA_ROOT/libero/libero"
BDDL_DIR="${BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_para}"
INIT_DIR="${INIT_DIR:-$LIBERO_INTERNAL_ROOT/init_files/libero_para}"
GOAL_BDDL_DIR="${GOAL_BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_goal}"

is_true() {
  case "${1,,}" in 1|true|yes|y|on) return 0 ;; *) return 1 ;; esac
}

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -f "$CONDA_SH" ]] || fail "Cannot find conda.sh: $CONDA_SH"
[[ -f "$CONDA_ENV/conda-meta/history" ]] || fail "Conda environment not found: $CONDA_ENV"
[[ -f "$EVAL_SCRIPT" ]] || fail "Evaluator not found: $EVAL_SCRIPT"
[[ -f "$SERVER_SCRIPT" ]] || fail "Server not found: $SERVER_SCRIPT"
[[ -f "$MODEL_PATH/config.json" ]] || fail "libero_goal checkpoint not found: $MODEL_PATH"
[[ -f "$MODEL_PATH/processor_config.json" ]] || fail "processor_config.json not found"
[[ -d "$COSMOS_PATH" ]] || fail "Local Cosmos model not found: $COSMOS_PATH"
[[ -d "$BDDL_DIR" && -d "$INIT_DIR" && -d "$GOAL_BDDL_DIR" ]] || fail "LIBERO-Para data is incomplete"
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] || fail "PORT must be positive"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
[[ "$N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "N_ACTION_STEPS must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || fail "NUM_STEPS_WAIT must be non-negative"
[[ "$INITIAL_STATE_INDEX" =~ ^[0-9]+$ ]] || fail "INITIAL_STATE_INDEX must be non-negative"
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_TASKS must be -1 or positive"
[[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || fail "NUM_SHARDS must be positive"
[[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || fail "SHARD_INDEX must be non-negative"
(( SHARD_INDEX < NUM_SHARDS )) || fail "SHARD_INDEX must be smaller than NUM_SHARDS"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="$(command -v python)"
[[ "$PYTHON_BIN" == "$CONDA_ENV/bin/python" ]] || fail "Activated unexpected Python: $PYTHON_BIN"

if [[ "$MODE" =~ ^(all|smoke|eval)$ ]] && {
  [[ -f "$OUTPUT_DIR/meta.json" ]] || [[ -d "$OUTPUT_DIR/episodes" ]];
}; then
  fail "Output already contains evaluation data: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR" "$RUNTIME_DIR/libero_config" "$RUNTIME_DIR/datasets"
export LIBERO_CONFIG_PATH="$RUNTIME_DIR/libero_config"
{
  printf 'benchmark_root: %s\n' "$LIBERO_INTERNAL_ROOT"
  printf 'bddl_files: %s\n' "$LIBERO_INTERNAL_ROOT/bddl_files"
  printf 'init_states: %s\n' "$LIBERO_INTERNAL_ROOT/init_files"
  printf 'assets: %s\n' "$LIBERO_INTERNAL_ROOT/assets"
  printf 'datasets: %s\n' "$RUNTIME_DIR/datasets"
} > "$LIBERO_CONFIG_PATH/config.yaml"

export PYTHONPATH="$LIBERO_PARA_ROOT:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES
export GR00T_EVAL_SEED="$SERVER_SEED"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
unset DISPLAY XAUTHORITY __GLX_VENDOR_LIBRARY_NAME || true

if [[ "$OFFLINE" == "1" ]]; then
  unset HF_ENDPOINT || true
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
INIT_COUNT="$(find "$INIT_DIR" -maxdepth 1 -type f -name 'eval*.pruned_init' -printf '.' | wc -c)"
GOAL_COUNT="$(find "$GOAL_BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
[[ "$BDDL_COUNT" == "4092" ]] || fail "Expected 4092 paraphrases, found $BDDL_COUNT"
[[ "$INIT_COUNT" == "10" && "$GOAL_COUNT" == "10" ]] || fail "Expected 10 init files and 10 goal BDDLs"

echo "[info] Mode: $MODE"
echo "[info] Global seed: $SEED"
echo "[info] Server seed: $SERVER_SEED"
echo "[info] Shard: $SHARD_INDEX/$NUM_SHARDS"
echo "[info] Checkpoint: $MODEL_PATH (libero_goal only)"
echo "[info] GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[info] LIBERO-Para: $BDDL_COUNT paraphrases over $GOAL_COUNT goal tasks"
echo "[info] Python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
echo "[info] Output: $OUTPUT_DIR"

run_preflight() {
  "$PYTHON_BIN" - "$MODEL_PATH" "$COSMOS_PATH" <<'PY'
import importlib
import json
import sys

model_path, cosmos_path = sys.argv[1:]
for module_name in (
    "torch",
    "transformers",
    "gr00t",
    "gr00t.policy.server_client",
    "libero.libero.envs",
    "mujoco",
    "zmq",
):
    importlib.import_module(module_name)

import torch

if not torch.cuda.is_available():
    raise SystemExit("GPU is unavailable in the gr00t_libero environment")
with open(f"{model_path}/config.json", "r", encoding="utf-8") as handle:
    config = json.load(handle)
configured_cosmos = config.get("model_name")
if configured_cosmos != cosmos_path:
    raise SystemExit(
        f"Checkpoint config model_name={configured_cosmos!r}; expected local Cosmos path {cosmos_path!r}. "
        "Refusing to mutate checkpoint files automatically."
    )
print(f"[check] torch={torch.__version__}")
print(f"[check] GPU 0={torch.cuda.get_device_name(0)}")
print(f"[check] checkpoint Cosmos path={configured_cosmos}")
PY
}

run_preflight
if [[ "$MODE" == "check" ]]; then
  echo "[ok] GR00T LIBERO-Para preflight passed for seed $SEED."
  exit 0
fi

port_is_listening() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5).close()
except OSError:
    raise SystemExit(1)
PY
}

assert_port_free() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
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

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[info] Stopping GR00T server pid=$SERVER_PID"
    kill -INT -- "-$SERVER_PID" >/dev/null 2>&1 || kill -INT "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 2
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      kill -- "-$SERVER_PID" >/dev/null 2>&1 || kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  assert_port_free
  echo "[info] Starting one seeded GR00T libero_goal server on $HOST:$PORT..."
  setsid "$PYTHON_BIN" -u "$SERVER_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag LIBERO_PANDA \
    --device cuda \
    --use-sim-policy-wrapper \
    --host "$HOST" \
    --port "$PORT" \
    --seed "$SERVER_SEED" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  echo "[info] Server pid=$SERVER_PID log=$SERVER_LOG"
}

wait_for_server() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  while ! port_is_listening; do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[error] GR00T server exited before becoming ready" >&2
      tail -n 160 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[error] Timed out waiting for GR00T server" >&2
      [[ -f "$SERVER_LOG" ]] && tail -n 160 "$SERVER_LOG" >&2
      exit 1
    fi
    sleep 2
  done
  echo "[info] GR00T server is ready."
}

run_evaluator() {
  local args=(
    --bddl-dir "$BDDL_DIR"
    --init-dir "$INIT_DIR"
    --goal-bddl-dir "$GOAL_BDDL_DIR"
    --host "$HOST"
    --port "$PORT"
    --seed "$SEED"
    --server-seed "$SERVER_SEED"
    --num-shards "$NUM_SHARDS"
    --shard-index "$SHARD_INDEX"
    --output-dir "$OUTPUT_DIR"
    --max-steps "$MAX_STEPS"
    --num-steps-wait "$NUM_STEPS_WAIT"
    --n-action-steps "$N_ACTION_STEPS"
    --initial-state-index "$INITIAL_STATE_INDEX"
    --client-timeout-ms "$CLIENT_TIMEOUT_MS"
  )
  [[ "$MAX_TASKS" != "-1" ]] && args+=(--max-tasks "$MAX_TASKS")
  is_true "$SAVE_VIDEO" && args+=(--save-video)
  is_true "$LOG_TRAJECTORIES" || args+=(--no-log-trajectories)
  (( ${#EXTRA_EVAL_ARGS[@]} == 0 )) || args+=("${EXTRA_EVAL_ARGS[@]}")

  set +e
  "$PYTHON_BIN" -u "$EVAL_SCRIPT" "${args[@]}" 2>&1 | tee "$EVAL_LOG"
  local status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

case "$MODE" in
  server)
    assert_port_free
    exec "$PYTHON_BIN" -u "$SERVER_SCRIPT" \
      --model-path "$MODEL_PATH" \
      --embodiment-tag LIBERO_PANDA \
      --device cuda \
      --use-sim-policy-wrapper \
      --host "$HOST" \
      --port "$PORT" \
      --seed "$SERVER_SEED"
    ;;
  eval)
    wait_for_server
    run_evaluator
    ;;
  all|smoke)
    start_server
    wait_for_server
    run_evaluator
    ;;
esac
