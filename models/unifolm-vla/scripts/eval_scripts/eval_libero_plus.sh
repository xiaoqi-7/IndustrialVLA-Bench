#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export PATH=$MACA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$MACA_HOME/lib:$MACA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOCAL_XVFB_ROOT="${LOCAL_XVFB_ROOT:-$BENCHMARK_ROOT/.local-xvfb/root}"
if [[ -x "$LOCAL_XVFB_ROOT/usr/bin/xvfb-run" ]]; then
  export PATH="$LOCAL_XVFB_ROOT/usr/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$LOCAL_XVFB_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-plus}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /root/envs/unifolm_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/unifolm_plus/bin/python
elif [[ -x /root/envs/unifolm_libero/bin/python ]]; then
  PYTHON_BIN=/root/envs/unifolm_libero/bin/python
else
  echo "[ERROR] Cannot find UnifoLM Python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

your_ckpt="${1:-${your_ckpt:-}}"                        # set to .../UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt
vlm_pretrained_path="${2:-${vlm_pretrained_path:-}}"    # set to a local UnifoLM-VLM-Base download
NUM_GPUS="${3:-${NUM_GPUS:-8}}"
LOG_BASE="${4:-${LOG_BASE:-$PROJECT_ROOT/logs2}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_GPUS must be a positive integer." >&2
  exit 1
fi
if [[ ! -f "$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json" ]]; then
  echo "[ERROR] LIBERO-plus task_classification.json not found under $LIBERO_PLUS_ROOT" >&2
  echo "[ERROR] Set LIBERO_PLUS_ROOT=/path/to/LIBERO-plus if needed." >&2
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
UNNORM_KEY="${UNNORM_KEY:-auto}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-UnifoLM-VLA}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
SAVE_FAILURE_VIDEO="${SAVE_FAILURE_VIDEO:-false}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
SUMMARY_ONLY="${SUMMARY_ONLY:-false}"
GPUS="${GPUS:-$(seq -s ' ' 0 $((NUM_GPUS - 1)))}"

read -r -a GPU_ARRAY <<< "$GPUS"
EVAL_WORLD_SIZE="${EVAL_WORLD_SIZE:-${WORLD_SIZE:-${#GPU_ARRAY[@]}}}"
if (( EVAL_WORLD_SIZE < 1 )); then
  echo "[ERROR] EVAL_WORLD_SIZE must be >= 1." >&2
  exit 1
fi
if (( ${#GPU_ARRAY[@]} < EVAL_WORLD_SIZE )); then
  echo "[ERROR] GPUS='$GPUS' has fewer entries than EVAL_WORLD_SIZE=$EVAL_WORLD_SIZE." >&2
  exit 1
fi

LOG_ROOT="$LOG_BASE/libero_plus_eval/$RUN_NAME"
EVAL_LOG_ROOT="$LOG_ROOT/eval"
RESULT_ROOT="$LOG_ROOT/results"
VIDEO_ROOT="$LOG_ROOT/videos"
SUMMARY_PATH="$LOG_ROOT/summary.md"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$LOG_ROOT/libero_config}"
CLASSIFICATION_PATH="$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"
LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"

mkdir -p "$EVAL_LOG_ROOT" "$RESULT_ROOT" "$VIDEO_ROOT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

export LIBERO_HOME="$LIBERO_PLUS_ROOT"
export LIBERO_CONFIG_PATH
export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_PLUS_ROOT}:${PYTHONPATH:-}"

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset MUJOCO_EGL_DEVICE_ID
unset EGL_DEVICE_ID
unset NVIDIA_VISIBLE_DEVICES
unset NVIDIA_DRIVER_CAPABILITIES
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

print_run_info() {
  echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
  echo "[INFO] LIBERO_PLUS_ROOT=$LIBERO_PLUS_ROOT"
  echo "[INFO] PYTHON=$PYTHON_BIN"
  echo "[INFO] checkpoint=$your_ckpt"
  echo "[INFO] vlm_pretrained_path=$vlm_pretrained_path"
  echo "[INFO] unnorm key: $UNNORM_KEY"
  echo "[INFO] task suites: $TASK_SUITES"
  echo "[INFO] GPUs: $GPUS, eval world size: $EVAL_WORLD_SIZE"
  echo "[INFO] logs: $LOG_ROOT"
}

write_summary() {
  local result_files=("$RESULT_ROOT"/rank_*.jsonl)
  if (( ${#result_files[@]} == 0 )); then
    echo "[ERROR] No result JSONL files found under $RESULT_ROOT." >&2
    return 1
  fi

  echo "[INFO] Writing LIBERO-plus summary from ${#result_files[@]} result file(s)."
  "$PYTHON_BIN" ./experiments/LIBERO/summarize_libero_plus.py "${result_files[@]}" --model "$MODEL_NAME" | tee "$SUMMARY_PATH"
}

print_run_info

case "${SUMMARY_ONLY,,}" in
  true|1|yes|y)
    write_summary
    echo "[INFO] Results: $RESULT_ROOT"
    echo "[INFO] Summary: $SUMMARY_PATH"
    exit 0
    ;;
esac

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "[ERROR] xvfb-run not found. Install xvfb/libgl/libglfw packages first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ["lazy_loader", "wand"] if importlib.util.find_spec(name) is None]
if missing:
    print(f"[ERROR] Missing Python package(s): {', '.join(missing)}", file=sys.stderr)
    print("[ERROR] Install in the UnifoLM env, e.g. pip install lazy_loader Wand", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as exc:
    print(f"[ERROR] Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available; UnifoLM-VLA LIBERO-plus eval must run on GPU.", file=sys.stderr)
    sys.exit(1)
print(f"[DEBUG] cuda available: {torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
PY

SAVE_FAILURE_VIDEO_ARGS=(--no-save-failure-video)
case "${SAVE_FAILURE_VIDEO,,}" in
  true|1|yes|y) SAVE_FAILURE_VIDEO_ARGS=(--save-failure-video) ;;
esac

PIDS=()
MONITOR_PID=""
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  if (( ${#PIDS[@]} > 0 )); then
    kill "${PIDS[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

for (( rank = 0; rank < EVAL_WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$rank]}"
  env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT CUDA_VISIBLE_DEVICES="$gpu" \
  xvfb-run -a -s "-screen 0 1024x768x24" \
  "$PYTHON_BIN" -X faulthandler ./experiments/LIBERO/eval_libero_plus.py \
    --pretrained-path "$your_ckpt" \
    --vlm-pretrained-path "$vlm_pretrained_path" \
    --unnorm-key "$UNNORM_KEY" \
    --window-size "$WINDOW_SIZE" \
    --libero-plus-root "$LIBERO_PLUS_ROOT" \
    --classification-path "$CLASSIFICATION_PATH" \
    --task-suites $TASK_SUITES \
    --eval-rank "$rank" \
    --eval-world-size "$EVAL_WORLD_SIZE" \
    --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
    --video-out-path "$VIDEO_ROOT/rank_${rank}" \
    --limit-tasks "$LIMIT_TASKS" \
    --result-jsonl "$RESULT_ROOT/rank_${rank}.jsonl" \
    "${SAVE_FAILURE_VIDEO_ARGS[@]}" \
    > "$EVAL_LOG_ROOT/eval_rank_${rank}_gpu_${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

echo "[INFO] Started $EVAL_WORLD_SIZE eval workers. Worker logs: $EVAL_LOG_ROOT"

case "${SHOW_PROGRESS,,}" in
  false|0|no|n) ;;
  *)
    "$PYTHON_BIN" ./experiments/LIBERO/monitor_libero_plus_progress.py \
      --result-dir "$RESULT_ROOT" \
      --classification-path "$CLASSIFICATION_PATH" \
      --task-suites $TASK_SUITES \
      --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
      --limit-tasks "$LIMIT_TASKS" &
    MONITOR_PID="$!"
    ;;
esac

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
PIDS=()

if [[ -n "$MONITOR_PID" ]]; then
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
  MONITOR_PID=""
fi

if (( status != 0 )); then
  echo "[ERROR] Some eval workers failed. Last log lines:" >&2
  for log_file in "$EVAL_LOG_ROOT"/*.log; do
    echo "===== $log_file =====" >&2
    tail -n 80 "$log_file" >&2 || true
  done
  exit 1
fi

write_summary

echo "[INFO] Results: $RESULT_ROOT"
echo "[INFO] Summary: $SUMMARY_PATH"
