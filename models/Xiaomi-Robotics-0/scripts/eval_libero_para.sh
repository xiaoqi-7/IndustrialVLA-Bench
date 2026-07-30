#!/usr/bin/env bash
set -Eeuo pipefail

# Xiaomi-Robotics-0 evaluation launcher for LIBERO-Para.
#
# Examples:
#   bash scripts/eval_libero_para.sh check
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_para.sh smoke
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_para.sh all
#
# The default "all" mode starts a local model server, waits until it is ready,
# runs all 4,092 LIBERO-Para instructions, and stops the server on exit.

usage() {
  cat <<'EOF'
Usage: eval_libero_para.sh [all|smoke|server|eval|check] [-- EVAL_ARG ...]

Modes:
  all      Start the model server, evaluate all tasks, then stop the server.
  smoke    Same as all, but evaluate one task (override with MAX_TASKS).
  server   Start only the model server in the foreground.
  eval     Run only the evaluator against an existing server.
  check    Validate paths, data, environment, imports, and GPU availability.

Common environment variables:
  CONDA_ENV              Conda prefix (default: /root/envs/xiaomi_libero)
  MODEL_PATH             Local Xiaomi checkpoint directory
  LIBERO_PARA_ROOT       LIBERO-Para repository directory
  CUDA_VISIBLE_DEVICES   GPU visible to the model server (default: 0)
  HOST / PORT            Server address (default: 127.0.0.1 / 10086)
  SEED                    Evaluation seed (default: 1; paper protocol runs 1, 7, 42)
  OUTPUT_DIR             Result directory (unique timestamped default)
  MAX_TASKS              Number of BDDL tasks; -1 means all (default: -1)
  MAX_STEPS              Maximum environment steps per task (default: 300)
  REPLAN_STEPS           Actions consumed from each model chunk (default: 10)
  NUM_STEPS_WAIT         Initial stabilization steps (default: 10)
  SAVE_VIDEO             true/false (default: false)
  RENDER_BACKEND         egl/osmesa (auto: osmesa on MetaX, otherwise egl)
  SERVER_READY_TIMEOUT_S Model loading timeout in seconds (default: 1800)

Arguments after "--" are appended to the official Python evaluator command.
EOF
}

MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
case "$MODE" in
  all|smoke|server|eval|check) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[error] Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_EVAL_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XIAOMI_ROOT="${XIAOMI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$XIAOMI_ROOT/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-para}"
CONDA_ENV="${CONDA_ENV:-/root/envs/xiaomi_libero}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
MODEL_PATH="${MODEL_PATH:-}"  # set to the Xiaomi-Robotics-0-LIBERO checkpoint directory

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10086}"
SEED="${SEED:-1}"  # paper protocol: three runs with SEED=1, 7, 42
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"

if [[ "$MODE" == "smoke" ]]; then
  MAX_TASKS="${MAX_TASKS:-1}"
else
  MAX_TASKS="${MAX_TASKS:--1}"
fi

RUN_NAME="${RUN_NAME:-xiaomi_robotics_0_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$XIAOMI_ROOT/results/libero_para/$RUN_NAME}"
SERVER_LOG="${SERVER_LOG:-$OUTPUT_DIR/server.log}"
EVAL_LOG="${EVAL_LOG:-$OUTPUT_DIR/eval.log}"
RUNTIME_DIR="${RUNTIME_DIR:-$OUTPUT_DIR/runtime}"

EVAL_SCRIPT="$LIBERO_PARA_ROOT/eval_scripts/examples/eval_xiaomi_robotics_0.py"
LIBERO_PARA_INTERNAL_ROOT="$LIBERO_PARA_ROOT/libero/libero"
BDDL_DIR="${BDDL_DIR:-$LIBERO_PARA_INTERNAL_ROOT/bddl_files/libero_para}"
INIT_DIR="${INIT_DIR:-$LIBERO_PARA_INTERNAL_ROOT/init_files/libero_para}"
GOAL_BDDL_DIR="${GOAL_BDDL_DIR:-$LIBERO_PARA_INTERNAL_ROOT/bddl_files/libero_goal}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[error] Required directory not found: $1" >&2
    exit 1
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] $name must be a positive integer, got: $value" >&2
    exit 1
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[error] $name must be a non-negative integer, got: $value" >&2
    exit 1
  fi
}

require_dir "$CONDA_ENV"
require_file "$CONDA_ENV/conda-meta/history"
require_file "$CONDA_SH"
require_dir "$XIAOMI_ROOT"
require_file "$XIAOMI_ROOT/deploy/server.py"
require_file "$XIAOMI_ROOT/deploy/client.py"
require_dir "$LIBERO_PARA_ROOT"
require_file "$EVAL_SCRIPT"
require_dir "$MODEL_PATH"
require_file "$MODEL_PATH/config.json"
require_dir "$BDDL_DIR"
require_dir "$INIT_DIR"
require_dir "$GOAL_BDDL_DIR"

require_positive_integer PORT "$PORT"
require_positive_integer MAX_STEPS "$MAX_STEPS"
require_nonnegative_integer NUM_STEPS_WAIT "$NUM_STEPS_WAIT"
require_positive_integer REPLAN_STEPS "$REPLAN_STEPS"
require_positive_integer SERVER_READY_TIMEOUT_S "$SERVER_READY_TIMEOUT_S"
if ! [[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] MAX_TASKS must be -1 or a positive integer, got: $MAX_TASKS" >&2
  exit 1
fi

# Activate the exact prefix requested for both server and client. This host's
# environment contains the GPU-enabled Torch build as well as LIBERO deps.
# Some environment activation hooks expand PYTHONPATH without a default while
# this launcher uses nounset, so ensure it exists before sourcing those hooks.
export PYTHONPATH="${PYTHONPATH:-}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="$(command -v python)"
if [[ "$PYTHON_BIN" != "$CONDA_ENV/bin/python" ]]; then
  echo "[error] Activated Python is $PYTHON_BIN, expected $CONDA_ENV/bin/python" >&2
  exit 1
fi

mkdir -p \
  "$OUTPUT_DIR" \
  "$RUNTIME_DIR/libero_config" \
  "$RUNTIME_DIR/hf_modules_cache" \
  "$RUNTIME_DIR/datasets"

# Do not use ~/.libero/config.yaml: it may point at an unrelated LIBERO clone.
LIBERO_CONFIG_PATH="$RUNTIME_DIR/libero_config"
export LIBERO_CONFIG_PATH
{
  printf 'benchmark_root: %s\n' "$LIBERO_PARA_INTERNAL_ROOT"
  printf 'bddl_files: %s\n' "$LIBERO_PARA_INTERNAL_ROOT/bddl_files"
  printf 'init_states: %s\n' "$LIBERO_PARA_INTERNAL_ROOT/init_files"
  printf 'assets: %s\n' "$LIBERO_PARA_INTERNAL_ROOT/assets"
  printf 'datasets: %s\n' "$RUNTIME_DIR/datasets"
} > "$LIBERO_CONFIG_PATH/config.yaml"

export PYTHONPATH="$LIBERO_PARA_ROOT:$XIAOMI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${RENDER_BACKEND:-}" ]]; then
  if command -v mx-smi >/dev/null 2>&1; then
    RENDER_BACKEND=osmesa
  else
    RENDER_BACKEND=egl
  fi
fi
case "$RENDER_BACKEND" in
  egl|osmesa) ;;
  *)
    echo "[error] RENDER_BACKEND must be egl or osmesa, got: $RENDER_BACKEND" >&2
    exit 1
    ;;
esac
# Keep MuJoCo and PyOpenGL on the same backend. The one-line setdefault change
# in the official evaluator lets this host use OSMesa while retaining EGL as
# its upstream default.
export MUJOCO_GL="$RENDER_BACKEND"
export PYOPENGL_PLATFORM="$RENDER_BACKEND"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_DIR/hf_modules_cache}"

BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
INIT_COUNT="$(find "$INIT_DIR" -maxdepth 1 -type f -name 'eval*.pruned_init' -printf '.' | wc -c)"
GOAL_COUNT="$(find "$GOAL_BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
if (( BDDL_COUNT == 0 )); then
  echo "[error] No BDDL tasks found under $BDDL_DIR" >&2
  exit 1
fi
if (( INIT_COUNT != 10 || GOAL_COUNT != 10 )); then
  echo "[error] Expected 10 init-state files and 10 goal BDDLs; found $INIT_COUNT and $GOAL_COUNT" >&2
  exit 1
fi

echo "[info] Mode: $MODE"
echo "[info] Conda environment: $CONDA_ENV"
echo "[info] Python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
echo "[info] Xiaomi root: $XIAOMI_ROOT"
echo "[info] LIBERO-Para root: $LIBERO_PARA_ROOT"
echo "[info] Checkpoint: $MODEL_PATH"
echo "[info] Dataset: $BDDL_COUNT paraphrases, $GOAL_COUNT base tasks"
echo "[info] Render backend: $RENDER_BACKEND"
echo "[info] Server: $HOST:$PORT"
echo "[info] Output: $OUTPUT_DIR"

run_preflight() {
  "$PYTHON_BIN" - "$MODEL_PATH" "$BDDL_DIR" "$INIT_DIR" "$GOAL_BDDL_DIR" <<'PY'
import importlib
import os
import sys

model_path, bddl_dir, init_dir, goal_bddl_dir = sys.argv[1:]
required = [
    "torch",
    "torchvision",
    "transformers",
    "flash_attn",
    "numpy",
    "PIL",
    "imageio",
    "tqdm",
    "libero.libero.envs",
    "deploy.client",
]
failed = {}
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = f"{type(exc).__name__}: {exc}"
if failed:
    details = "; ".join(f"{name} ({error})" for name, error in failed.items())
    raise SystemExit(f"[error] Failed to import required Python modules: {details}")

import torch
import transformers
from libero.libero import get_libero_path

if not torch.cuda.is_available():
    raise SystemExit("[error] torch.cuda.is_available() is false; model inference requires a GPU")
if os.path.realpath(get_libero_path("benchmark_root")) != os.path.realpath(
    os.path.dirname(os.path.dirname(bddl_dir))
):
    raise SystemExit("[error] LIBERO_CONFIG_PATH does not point to the selected LIBERO-Para tree")

print(f"[check] torch={torch.__version__}, transformers={transformers.__version__}")
print(f"[check] CUDA devices visible={torch.cuda.device_count()}, device0={torch.cuda.get_device_name(0)}")
print(f"[check] model={model_path}")
print(f"[check] bddl_dir={bddl_dir}")
print(f"[check] init_dir={init_dir}")
print(f"[check] goal_bddl_dir={goal_bddl_dir}")
PY
}

run_preflight
if [[ "$MODE" == "check" ]]; then
  echo "[ok] LIBERO-Para evaluation preflight passed."
  exit 0
fi

port_is_listening() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

assert_port_is_free() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((sys.argv[1], int(sys.argv[2])))
except OSError as exc:
    raise SystemExit(f"[error] Cannot bind {sys.argv[1]}:{sys.argv[2]}: {exc}")
finally:
    sock.close()
PY
}

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[info] Stopping model server (pid $SERVER_PID)..."
    kill -INT -- "-$SERVER_PID" >/dev/null 2>&1 || kill -INT "$SERVER_PID" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      kill -- "-$SERVER_PID" >/dev/null 2>&1 || kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  assert_port_is_free
  echo "[info] Starting Xiaomi model server on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    setsid "$PYTHON_BIN" "$XIAOMI_ROOT/deploy/server.py" \
      --model "$MODEL_PATH" \
      --host "$HOST" \
      --port "$PORT" \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  echo "[info] Model server pid: $SERVER_PID"
  echo "[info] Model server log: $SERVER_LOG"
}

wait_for_server() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  echo "[info] Waiting up to ${SERVER_READY_TIMEOUT_S}s for the model server..."
  while ! port_is_listening; do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[error] Model server exited before becoming ready." >&2
      tail -n 120 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[error] Timed out waiting for $HOST:$PORT" >&2
      if [[ -f "$SERVER_LOG" ]]; then
        tail -n 120 "$SERVER_LOG" >&2 || true
      fi
      exit 1
    fi
    sleep 2
  done
  echo "[info] Model server is ready."
}

run_evaluator() {
  if [[ -f "$OUTPUT_DIR/meta.json" || -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "[error] Output already contains evaluation data: $OUTPUT_DIR" >&2
    echo "[error] Select a new OUTPUT_DIR to avoid duplicated episode records." >&2
    exit 1
  fi

  local eval_args=(
    --bddl_dir "$BDDL_DIR"
    --init_dir "$INIT_DIR"
    --goal_bddl_dir "$GOAL_BDDL_DIR"
    --host "$HOST"
    --port "$PORT"
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
    --max_steps "$MAX_STEPS"
    --num_steps_wait "$NUM_STEPS_WAIT"
    --replan_steps "$REPLAN_STEPS"
    --mode para
  )
  if [[ "$MAX_TASKS" != "-1" ]]; then
    eval_args+=(--max_tasks "$MAX_TASKS")
  fi
  if is_true "$SAVE_VIDEO"; then
    eval_args+=(--save_video)
  fi
  if (( ${#EXTRA_EVAL_ARGS[@]} > 0 )); then
    eval_args+=("${EXTRA_EVAL_ARGS[@]}")
  fi

  echo "[info] Starting evaluator (MAX_TASKS=$MAX_TASKS, MAX_STEPS=$MAX_STEPS)..."
  set +e
  "$PYTHON_BIN" -u "$EVAL_SCRIPT" "${eval_args[@]}" 2>&1 | tee "$EVAL_LOG"
  local status="${PIPESTATUS[0]}"
  set -e
  if (( status != 0 )); then
    echo "[error] Evaluator failed with exit code $status. Log: $EVAL_LOG" >&2
    return "$status"
  fi
  echo "[ok] Evaluation finished."
  echo "[ok] Summary: $OUTPUT_DIR/summary.json"
  echo "[ok] Full log: $EVAL_LOG"
}

case "$MODE" in
  server)
    assert_port_is_free
    echo "[info] Starting foreground model server on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}..."
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PYTHON_BIN" "$XIAOMI_ROOT/deploy/server.py" \
        --model "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT"
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
