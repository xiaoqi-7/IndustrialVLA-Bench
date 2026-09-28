#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Cloud-server LIBERO-plus eval for UnifoLM-VLA.
# This file intentionally uses absolute paths only, so it can be launched from any cwd.

PROJECT_ROOT="/mnt/afs/zhengmingkai/raozf/benchmark/unifolm-vla"
BENCHMARK_ROOT="/mnt/afs/zhengmingkai/raozf/benchmark"
LIBERO_PLUS_ROOT="/mnt/afs/zhengmingkai/raozf/benchmark/LIBERO-plus"
PYTHON_BIN="/root/envs/unifolm_plus/bin/python"

DEFAULT_CKPT="/mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt"
DEFAULT_VLM="/mnt/afs/zhengmingkai/raozf/models/UnifoLM/UnifoLM-VLM-Base"

your_ckpt="${1:-${your_ckpt:-$DEFAULT_CKPT}}"
vlm_pretrained_path="${2:-${vlm_pretrained_path:-$DEFAULT_VLM}}"
NUM_GPUS="${3:-${NUM_GPUS:-1}}"
LOG_BASE="${4:-${LOG_BASE:-$PROJECT_ROOT/logs}}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$LIBERO_PLUS_ROOT" ]]; then
  echo "[ERROR] LIBERO_PLUS_ROOT not found: $LIBERO_PLUS_ROOT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN" >&2
  echo "[ERROR] Please create/use /root/envs/unifolm_plus, or edit PYTHON_BIN in this script." >&2
  exit 1
fi
if [[ ! -f "$your_ckpt" ]]; then
  echo "[ERROR] Checkpoint not found: $your_ckpt" >&2
  exit 1
fi
if [[ ! -e "$vlm_pretrained_path" ]]; then
  echo "[ERROR] VLM path not found: $vlm_pretrained_path" >&2
  exit 1
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_GPUS must be a positive integer." >&2
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
UNNORM_KEY="${UNNORM_KEY:-libero_object_no_noops}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-UnifoLM-VLA}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
SAVE_FAILURE_VIDEO="${SAVE_FAILURE_VIDEO:-false}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
GPUS="${GPUS:-$(seq -s ' ' 0 $((NUM_GPUS - 1)))}"

read -r -a GPU_ARRAY <<< "$GPUS"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ARRAY[@]}}"
if (( WORLD_SIZE < 1 )); then
  echo "[ERROR] WORLD_SIZE must be >= 1." >&2
  exit 1
fi
if (( ${#GPU_ARRAY[@]} < WORLD_SIZE )); then
  echo "[ERROR] GPUS='$GPUS' has fewer entries than WORLD_SIZE=$WORLD_SIZE." >&2
  exit 1
fi

LOG_ROOT="$LOG_BASE/libero_plus_eval/$RUN_NAME"
EVAL_LOG_ROOT="$LOG_ROOT/eval"
RESULT_ROOT="$LOG_ROOT/results"
VIDEO_ROOT="$LOG_ROOT/videos"
SUMMARY_PATH="$LOG_ROOT/summary.md"
LIBERO_CONFIG_PATH="$LOG_ROOT/libero_config"
CLASSIFICATION_PATH="$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json"
LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"

if [[ ! -f "$CLASSIFICATION_PATH" ]]; then
  echo "[ERROR] LIBERO-plus task_classification.json not found: $CLASSIFICATION_PATH" >&2
  exit 1
fi

mkdir -p "$EVAL_LOG_ROOT" "$RESULT_ROOT" "$VIDEO_ROOT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export PATH=$MACA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$MACA_HOME/lib:$MACA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export LIBERO_HOME="$LIBERO_PLUS_ROOT"
export LIBERO_CONFIG_PATH
export PYTHONPATH="$PROJECT_ROOT:$LIBERO_PLUS_ROOT:${PYTHONPATH:-}"

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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Headless cloud rendering: no desktop is needed.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset DISPLAY
unset XAUTHORITY
unset MUJOCO_EGL_DEVICE_ID
unset EGL_DEVICE_ID
unset NVIDIA_VISIBLE_DEVICES
unset NVIDIA_DRIVER_CAPABILITIES
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "[ERROR] xvfb-run not found. Install on cloud server:" >&2
  echo "        apt-get update && apt-get install -y xvfb libgl1-mesa-glx libgl1-mesa-dri libglfw3 mesa-utils" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ["lazy_loader", "wand"] if importlib.util.find_spec(name) is None]
if missing:
    print(f"[ERROR] Missing Python package(s): {', '.join(missing)}", file=sys.stderr)
    print("[ERROR] Install in /root/envs/unifolm_plus, e.g. pip install lazy_loader Wand", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as exc:
    print(f"[ERROR] Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available; cloud eval must run on GPU.", file=sys.stderr)
    sys.exit(1)
print(f"[DEBUG] cuda available: {torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
PY

SAVE_FAILURE_VIDEO_ARGS=(--no-save-failure-video)
case "${SAVE_FAILURE_VIDEO,,}" in
  true|1|yes|y) SAVE_FAILURE_VIDEO_ARGS=(--save-failure-video) ;;
esac

cd "$PROJECT_ROOT"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] LIBERO_PLUS_ROOT=$LIBERO_PLUS_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] checkpoint=$your_ckpt"
echo "[INFO] vlm_pretrained_path=$vlm_pretrained_path"
echo "[INFO] task suites: $TASK_SUITES"
echo "[INFO] GPUs: $GPUS, world size: $WORLD_SIZE"
echo "[INFO] logs: $LOG_ROOT"

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

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$rank]}"
  CUDA_VISIBLE_DEVICES="$gpu" RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
  xvfb-run -a -s "-screen 0 1024x768x24" \
  "$PYTHON_BIN" -X faulthandler "$PROJECT_ROOT/experiments/LIBERO/eval_libero_plus.py" \
    --pretrained-path "$your_ckpt" \
    --vlm-pretrained-path "$vlm_pretrained_path" \
    --unnorm-key "$UNNORM_KEY" \
    --window-size "$WINDOW_SIZE" \
    --libero-plus-root "$LIBERO_PLUS_ROOT" \
    --classification-path "$CLASSIFICATION_PATH" \
    --task-suites $TASK_SUITES \
    --eval-rank "$rank" \
    --eval-world-size "$WORLD_SIZE" \
    --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
    --video-out-path "$VIDEO_ROOT/rank_${rank}" \
    --limit-tasks "$LIMIT_TASKS" \
    --result-jsonl "$RESULT_ROOT/rank_${rank}.jsonl" \
    "${SAVE_FAILURE_VIDEO_ARGS[@]}" \
    > "$EVAL_LOG_ROOT/eval_rank_${rank}_gpu_${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

echo "[INFO] Started $WORLD_SIZE eval workers. Worker logs: $EVAL_LOG_ROOT"

case "${SHOW_PROGRESS,,}" in
  false|0|no|n) ;;
  *)
    "$PYTHON_BIN" "$PROJECT_ROOT/experiments/LIBERO/monitor_libero_plus_progress.py" \
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
  if (( status == 0 )); then
    wait "$MONITOR_PID" || true
  else
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
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

RESULT_FILES=("$RESULT_ROOT"/rank_*.jsonl)
if (( ${#RESULT_FILES[@]} == 0 )); then
  echo "[ERROR] No result JSONL files found under $RESULT_ROOT." >&2
  exit 1
fi

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/LIBERO/summarize_libero_plus.py" "${RESULT_FILES[@]}" --model "$MODEL_NAME" | tee "$SUMMARY_PATH"

echo "[INFO] Results: $RESULT_ROOT"
echo "[INFO] Summary: $SUMMARY_PATH"
