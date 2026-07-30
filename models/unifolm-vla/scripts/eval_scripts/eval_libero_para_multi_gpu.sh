#!/usr/bin/env bash
set -Eeuo pipefail

# Parallel UnifoLM-VLA evaluation for LIBERO-Para. Each worker loads one model
# replica on one physical GPU and evaluates a deterministic round-robin shard.

usage() {
  cat <<'EOF'
Usage: eval_libero_para_multi_gpu.sh [check] [-- EVAL_ARG ...]

Examples:
  GPUS="0 1 2 3 4 5 6 7" \
    bash scripts/eval_libero_para_multi_gpu.sh

  GPUS=0 MAX_TASKS=1 SAVE_VIDEO=true \
    bash scripts/eval_libero_para_multi_gpu.sh

  bash scripts/eval_libero_para_multi_gpu.sh check

Important environment variables:
  CONDA_ENV             Python environment prefix
                        (default: /root/envs/unifolm-libero)
  GPUS                  Space/comma-separated physical GPU IDs
                        (default: CUDA_VISIBLE_DEVICES or 0)
  NUM_WORKERS           Number of GPU workers (default: number of GPUS entries)
  PRETRAINED_PATH       UnifoLM-VLA LIBERO pytorch_model.pt checkpoint
  VLM_PRETRAINED_PATH   Local UnifoLM-VLM-Base directory
  LIBERO_PARA_ROOT      LIBERO-Para checkout (auto-detected by default)
  OUTPUT_DIR            Root result directory (timestamped default)
  MAX_TASKS             Global task limit before sharding; -1 runs all 4092
  MAX_STEPS             Maximum policy steps per paraphrase (default: 300)
  NUM_STEPS_WAIT        Initial no-op stabilization steps (default: 10)
  REPLAN_STEPS          Actions consumed per predicted chunk (default: 8)
  WINDOW_SIZE           Observation-history size (default: 2)
  UNNORM_KEY            Action/state statistics key
                        (default: libero_goal_no_noops)
  SEED                  Evaluation seed (default: 1; paper protocol runs 1, 7, 42)
  SAVE_VIDEO            true/false (default: false)
  RENDER_BACKEND        egl/osmesa (auto-detected default)
  USE_XVFB              auto/true/false (default: auto)
  MONITOR_INTERVAL_S    Aggregate progress refresh interval (default: 2)
  CHECK_ONLY            true validates the setup without loading the model

The merged output follows the LIBERO-Para layout:
  OUTPUT_DIR/{eval0..9,meta,summary}.json
Worker results and logs remain under OUTPUT_DIR/workers and OUTPUT_DIR/logs.
Arguments after "--" are appended to every Python evaluator invocation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "check" ]]; then
  CHECK_ONLY=true
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_EVAL_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-para}"

CONDA_ENV="${CONDA_ENV:-/root/envs/unifolm_libero}"
PYTHON_BIN="$CONDA_ENV/bin/python"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"          # set to .../UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-}"  # set to a local UnifoLM-VLM-Base download
UNNORM_KEY="${UNNORM_KEY:-libero_goal_no_noops}"

EVAL_SCRIPT="$PROJECT_ROOT/experiments/LIBERO/eval_libero_para.py"
MONITOR_SCRIPT="$PROJECT_ROOT/experiments/LIBERO/monitor_libero_para_progress.py"
MERGE_SCRIPT="$PROJECT_ROOT/experiments/LIBERO/merge_libero_para_results.py"
LIBERO_INTERNAL_ROOT="$LIBERO_PARA_ROOT/libero/libero"
BDDL_DIR="${BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_para}"
INIT_DIR="${INIT_DIR:-$LIBERO_INTERNAL_ROOT/init_files/libero_para}"
GOAL_BDDL_DIR="${GOAL_BDDL_DIR:-$LIBERO_INTERNAL_ROOT/bddl_files/libero_goal}"

SEED="${SEED:-1}"  # paper protocol: three runs with SEED=1, 7, 42
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-2}"
CHECK_ONLY="${CHECK_ONLY:-false}"

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

require_file() {
  [[ -f "$1" ]] || fail "Required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "Required directory not found: $1"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer: $value"
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be a non-negative integer: $value"
}

[[ -x "$PYTHON_BIN" ]] || fail \
  "Python not found at $PYTHON_BIN. Create the requested environment or set CONDA_ENV."
require_file "$EVAL_SCRIPT"
require_file "$MONITOR_SCRIPT"
require_file "$MERGE_SCRIPT"
require_file "$PRETRAINED_PATH"
require_dir "$VLM_PRETRAINED_PATH"
require_file "$VLM_PRETRAINED_PATH/config.json"
require_dir "$LIBERO_PARA_ROOT"
require_dir "$BDDL_DIR"
require_dir "$INIT_DIR"
require_dir "$GOAL_BDDL_DIR"

MODEL_RUN_DIR="$(dirname "$(dirname "$PRETRAINED_PATH")")"
require_file "$MODEL_RUN_DIR/config.yaml"
require_file "$MODEL_RUN_DIR/dataset_statistics.json"

require_positive_integer MAX_STEPS "$MAX_STEPS"
require_nonnegative_integer NUM_STEPS_WAIT "$NUM_STEPS_WAIT"
require_positive_integer REPLAN_STEPS "$REPLAN_STEPS"
require_positive_integer WINDOW_SIZE "$WINDOW_SIZE"
if [[ "$MAX_TASKS" != "-1" && ! "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  fail "MAX_TASKS must be -1 or a positive integer: $MAX_TASKS"
fi

GPU_TEXT="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
GPU_TEXT="${GPU_TEXT//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_TEXT"
(( ${#GPU_ARRAY[@]} > 0 )) || fail "GPUS must contain at least one GPU ID"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ARRAY[@]}}"
require_positive_integer NUM_WORKERS "$NUM_WORKERS"
(( NUM_WORKERS <= ${#GPU_ARRAY[@]} )) || \
  fail "NUM_WORKERS exceeds the number of GPUS entries"

declare -A SEEN_GPUS=()
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU ID must be a non-negative integer: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || fail "Duplicate GPU ID: $gpu"
  SEEN_GPUS[$gpu]=1
done

BDDL_COUNT="$(find "$BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
INIT_COUNT="$(find "$INIT_DIR" -maxdepth 1 -type f -name 'eval*.pruned_init' -printf '.' | wc -c)"
GOAL_COUNT="$(find "$GOAL_BDDL_DIR" -maxdepth 1 -type f -name '*.bddl' -printf '.' | wc -c)"
(( BDDL_COUNT > 0 )) || fail "No LIBERO-Para BDDL files found under $BDDL_DIR"
(( INIT_COUNT == 10 )) || fail "Expected 10 init-state files, found $INIT_COUNT"
(( GOAL_COUNT == 10 )) || fail "Expected 10 LIBERO-Goal BDDLs, found $GOAL_COUNT"

if (( MAX_TASKS > 0 && MAX_TASKS < BDDL_COUNT )); then
  EXPECTED_TOTAL="$MAX_TASKS"
else
  EXPECTED_TOTAL="$BDDL_COUNT"
fi
(( NUM_WORKERS <= EXPECTED_TOTAL )) || fail "More workers than selected tasks"

RUN_NAME="${RUN_NAME:-unifolm_vla_seed${SEED}_${NUM_WORKERS}gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/libero_para/$RUN_NAME}"
WORKERS_DIR="$OUTPUT_DIR/workers"
LOG_DIR="$OUTPUT_DIR/logs"
RUNTIME_DIR="$OUTPUT_DIR/runtime"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RUNTIME_DIR/libero_config}"

if ! is_true "$CHECK_ONLY" && \
   [[ -d "$WORKERS_DIR" || -f "$OUTPUT_DIR/eval0.json" || \
      -f "$OUTPUT_DIR/meta.json" || -f "$OUTPUT_DIR/summary.json" ]]; then
  fail "Output directory already contains parallel evaluation data: $OUTPUT_DIR"
fi

# The available UnifoLM build targets MetaX. Include both runtime and OpenMP
# libraries explicitly so invoking the environment's Python works in a clean shell.
if [[ -d /opt/maca ]]; then
  MACA_HOME="${MACA_HOME:-/opt/maca}"
  MACA_PATH="${MACA_PATH:-$MACA_HOME}"
  export MACA_HOME MACA_PATH
  export PATH="$MACA_HOME/bin:/opt/mxdriver/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$MACA_HOME/lib:$MACA_HOME/lib64:$MACA_HOME/mxgpu_llvm/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

LOCAL_XVFB_ROOT="${LOCAL_XVFB_ROOT:-$BENCHMARK_ROOT/.local-xvfb/root}"
if [[ -x "$LOCAL_XVFB_ROOT/usr/bin/xvfb-run" ]]; then
  export PATH="$LOCAL_XVFB_ROOT/usr/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$LOCAL_XVFB_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -z "${RENDER_BACKEND:-}" ]]; then
  if command -v mx-smi >/dev/null 2>&1; then
    RENDER_BACKEND=osmesa
  else
    RENDER_BACKEND=egl
  fi
fi
case "$RENDER_BACKEND" in
  egl|osmesa) ;;
  *) fail "RENDER_BACKEND must be egl or osmesa: $RENDER_BACKEND" ;;
esac
export MUJOCO_GL="$RENDER_BACKEND"
export PYOPENGL_PLATFORM="$RENDER_BACKEND"
if [[ "$RENDER_BACKEND" == "osmesa" ]]; then
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"
fi

USE_XVFB="${USE_XVFB:-auto}"
RUN_WITH_XVFB=false
case "${USE_XVFB,,}" in
  auto)
    if [[ "$RENDER_BACKEND" == "osmesa" ]] && command -v xvfb-run >/dev/null 2>&1; then
      RUN_WITH_XVFB=true
    fi
    ;;
  1|true|yes|y|on)
    command -v xvfb-run >/dev/null 2>&1 || fail "USE_XVFB=true but xvfb-run was not found"
    RUN_WITH_XVFB=true
    ;;
  0|false|no|n|off) ;;
  *) fail "USE_XVFB must be auto, true, or false: $USE_XVFB" ;;
esac

export LIBERO_HOME="$LIBERO_PARA_ROOT"
export LIBERO_CONFIG_PATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$LIBERO_PARA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_DIR/hf_modules_cache}"

mkdir -p \
  "$RUNTIME_DIR/datasets" \
  "$LIBERO_CONFIG_PATH" \
  "$HF_MODULES_CACHE"
if ! is_true "$CHECK_ONLY"; then
  mkdir -p "$WORKERS_DIR" "$LOG_DIR/workers"
fi

# Never use ~/.libero/config.yaml; it may point at a different LIBERO clone.
{
  printf 'benchmark_root: %s\n' "$LIBERO_INTERNAL_ROOT"
  printf 'bddl_files: %s\n' "$LIBERO_INTERNAL_ROOT/bddl_files"
  printf 'init_states: %s\n' "$LIBERO_INTERNAL_ROOT/init_files"
  printf 'assets: %s\n' "$LIBERO_INTERNAL_ROOT/assets"
  printf 'datasets: %s\n' "$RUNTIME_DIR/datasets"
} > "$LIBERO_CONFIG_PATH/config.yaml"

echo "[info] UnifoLM-VLA multi-GPU LIBERO-Para evaluation"
echo "[info] Environment: $CONDA_ENV"
echo "[info] Python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
echo "[info] Checkpoint: $PRETRAINED_PATH"
echo "[info] VLM: $VLM_PRETRAINED_PATH"
echo "[info] LIBERO-Para: $LIBERO_PARA_ROOT"
echo "[info] GPUs: ${GPU_ARRAY[*]:0:$NUM_WORKERS}"
echo "[info] Workers: $NUM_WORKERS"
echo "[info] Tasks: $EXPECTED_TOTAL (round-robin sharding)"
echo "[info] Render backend: $RENDER_BACKEND (xvfb=$RUN_WITH_XVFB)"
echo "[info] Output: $OUTPUT_DIR"

run_preflight() {
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON_BIN" - \
    "$PRETRAINED_PATH" \
    "$VLM_PRETRAINED_PATH" \
    "$UNNORM_KEY" \
    "$LIBERO_INTERNAL_ROOT" \
    "$BDDL_COUNT" \
    "$REPLAN_STEPS" <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path

checkpoint, vlm_path, unnorm_key, libero_root, bddl_count, replan_steps = sys.argv[1:]
required = [
    "torch",
    "tensorflow",
    "qwen_vl_utils",
    "imageio",
    "libero.libero.envs",
    "unifolm_vla",
    "experiments.LIBERO.eval_libero_para",
]
failures = {}
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures[name] = f"{type(exc).__name__}: {exc}"
if failures:
    details = "; ".join(f"{name} ({error})" for name, error in failures.items())
    raise SystemExit(f"[error] Failed to import required modules: {details}")

import torch
from libero.libero import get_libero_path
from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK

if not torch.cuda.is_available():
    raise SystemExit("[error] torch.cuda.is_available() is false")
if int(replan_steps) > NUM_ACTIONS_CHUNK:
    raise SystemExit(
        f"[error] REPLAN_STEPS={replan_steps} exceeds model action chunk "
        f"length NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
    )
if os.path.realpath(get_libero_path("benchmark_root")) != os.path.realpath(libero_root):
    raise SystemExit("[error] LIBERO_CONFIG_PATH does not point at LIBERO-Para")

run_dir = Path(checkpoint).parents[1]
with (run_dir / "dataset_statistics.json").open("r", encoding="utf-8") as handle:
    stats = json.load(handle)
if unnorm_key not in stats:
    raise SystemExit(
        f"[error] UNNORM_KEY={unnorm_key!r} is unavailable; choose from {sorted(stats)}"
    )

print(f"[check] torch={torch.__version__}")
print(
    f"[check] CUDA devices visible={torch.cuda.device_count()}, "
    f"device0={torch.cuda.get_device_name(0)}"
)
print(f"[check] action chunk={NUM_ACTIONS_CHUNK}, replan steps={replan_steps}")
print(f"[check] LIBERO-Para BDDLs={bddl_count}")
print(f"[check] checkpoint={checkpoint}")
print(f"[check] VLM={vlm_path}")
PY
}

run_preflight
if is_true "$CHECK_ONLY"; then
  echo "[ok] UnifoLM-VLA LIBERO-Para preflight passed."
  exit 0
fi

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
  if (( ${#EVAL_PIDS[@]} > 0 )); then
    sleep 1
  fi
  for pid in "${EVAL_PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

common_args=(
  --pretrained-path "$PRETRAINED_PATH"
  --vlm-pretrained-path "$VLM_PRETRAINED_PATH"
  --unnorm-key "$UNNORM_KEY"
  --bddl-dir "$BDDL_DIR"
  --init-dir "$INIT_DIR"
  --goal-bddl-dir "$GOAL_BDDL_DIR"
  --seed "$SEED"
  --max-steps "$MAX_STEPS"
  --num-steps-wait "$NUM_STEPS_WAIT"
  --window-size "$WINDOW_SIZE"
  --replan-steps "$REPLAN_STEPS"
  --max-tasks "$MAX_TASKS"
)
if is_true "$SAVE_VIDEO"; then
  common_args+=(--save-video)
fi

for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  gpu="${GPU_ARRAY[$rank]}"
  worker_output="$WORKERS_DIR/rank_${rank}"
  worker_log="$LOG_DIR/workers/rank_${rank}_gpu_${gpu}.log"
  mkdir -p "$worker_output"

  runner=("$PYTHON_BIN" -X faulthandler "$EVAL_SCRIPT")
  if [[ "$RUN_WITH_XVFB" == "true" ]]; then
    runner=(xvfb-run -a -s "-screen 0 1024x768x24" "${runner[@]}")
  fi

  setsid env \
    -u WORLD_SIZE \
    -u RANK \
    -u LOCAL_RANK \
    -u MASTER_ADDR \
    -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "${runner[@]}" \
    "${common_args[@]}" \
    --output-dir "$worker_output" \
    --num-shards "$NUM_WORKERS" \
    --shard-index "$rank" \
    "${EXTRA_EVAL_ARGS[@]}" \
    >"$worker_log" 2>&1 &
  EVAL_PIDS+=("$!")
done

echo "[info] Started $NUM_WORKERS evaluation workers."
echo "[info] Worker logs: $LOG_DIR/workers"
"$PYTHON_BIN" -u "$MONITOR_SCRIPT" \
  --workers-dir "$WORKERS_DIR" \
  --total "$EXPECTED_TOTAL" \
  --interval "$MONITOR_INTERVAL_S" &
MONITOR_PID="$!"

status=0
for ((rank = 0; rank < NUM_WORKERS; rank++)); do
  if ! wait "${EVAL_PIDS[$rank]}"; then
    echo "[error] Evaluation rank $rank failed." >&2
    status=1
  fi
done
EVAL_PIDS=()

if (( status != 0 )); then
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=""
  fi
  for log_file in "$LOG_DIR/workers"/*.log; do
    echo "===== $log_file =====" >&2
    tail -n 100 "$log_file" >&2 || true
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
