#!/usr/bin/env bash
# Evaluate the local FastWAM LIBERO checkpoint with seeds 1, 7 and 42.
#
# Examples:
#   bash FastWAM/run_fastwam_libero.sh
#   GPUS="0,1,2,3" NUM_TRIALS_PER_TASK=1 MAX_TASKS=4 \
#     bash FastWAM/run_fastwam_libero.sh
#   TASK_SUITES="libero_spatial libero_goal" GPUS="4 5" \
#     bash FastWAM/run_fastwam_libero.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/root/envs/fastwam_libero/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"  # set to the FastWAM LIBERO checkpoint directory
LIBERO_ROOT="${LIBERO_ROOT:-$BENCHMARK_ROOT/LIBERO}"
FASTWAM_SOURCE_ROOT="${FASTWAM_SOURCE_ROOT:-$BENCHMARK_ROOT/FastWAM}"
LEROBOT_ZIP="${LEROBOT_ZIP:-$BENCHMARK_ROOT/unifolm-vla/lerobot.zip}"

SEEDS_TEXT="${SEEDS:-1 7 42}"
TASK_SUITES_TEXT="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-30}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TASKS="${MAX_TASKS:--1}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/result/libero_3seeds/$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<'EOF'
Usage: bash FastWAM/run_fastwam_libero.sh

Runs LIBERO for seeds 1, 7 and 42, using one worker per selected GPU, and
writes three_seed_summary.json plus four_task_summary.json.  Configuration is supplied through environment
variables, for example:

  GPUS="0,1,2,3" NUM_TRIALS_PER_TASK=50 bash FastWAM/run_fastwam_libero.sh
  TASK_SUITES="libero_spatial libero_goal" MAX_TASKS=2 NUM_TRIALS_PER_TASK=1 \
    DRY_RUN=1 bash FastWAM/run_fastwam_libero.sh

Important variables: MODEL_PATH, LIBERO_ROOT, PYTHON_BIN, GPUS, SEEDS,
TASK_SUITES, NUM_TRIALS_PER_TASK, RESULT_ROOT, MUJOCO_GL.
EOF
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

die() {
  echo "[error] $*" >&2
  exit 1
}

info() {
  echo "[info] $*"
}

[[ -x "$PYTHON_BIN" ]] || die "Python not found or not executable: $PYTHON_BIN"
[[ -d "$MODEL_PATH" ]] || die "Model directory not found: $MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || die "Missing model config: $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/model.safetensors" ]] || die "Missing checkpoint: $MODEL_PATH/model.safetensors"
[[ -f "$MODEL_PATH/models_t5_umt5-xxl-enc-bf16.safetensors" ]] || die "Missing UMT5 sidecar"
[[ -f "$MODEL_PATH/Wan2.2_VAE.safetensors" ]] || die "Missing Wan VAE sidecar"
[[ -d "$MODEL_PATH/google/umt5-xxl" ]] || die "Missing local UMT5 tokenizer"
[[ -d "$LIBERO_ROOT/libero/libero/bddl_files" ]] || die "LIBERO bddl_files not found under $LIBERO_ROOT"
[[ -d "$LIBERO_ROOT/libero/libero/init_files" ]] || die "LIBERO init_files not found under $LIBERO_ROOT"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT//,/ }"
(( ${#SEED_ARRAY[@]} == 3 )) || die "Exactly three seeds are required; got: ${SEED_ARRAY[*]}"
declare -A SEEN_SEEDS=()
for seed in "${SEED_ARRAY[@]}"; do
  [[ "$seed" =~ ^-?[0-9]+$ ]] || die "Invalid seed: $seed"
  [[ -z "${SEEN_SEEDS[$seed]:-}" ]] || die "Duplicate seed: $seed"
  SEEN_SEEDS[$seed]=1
done

read -r -a SUITE_ARRAY <<< "${TASK_SUITES_TEXT//,/ }"
(( ${#SUITE_ARRAY[@]} > 0 )) || die "TASK_SUITES is empty"
[[ "$NUM_TRIALS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || die "NUM_TRIALS_PER_TASK must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || die "NUM_STEPS_WAIT must be non-negative"
[[ "$REPLAN_STEPS" =~ ^[1-9][0-9]*$ ]] || die "REPLAN_STEPS must be positive"
[[ "$ACTION_HORIZON" =~ ^[0-9]+$ ]] || die "ACTION_HORIZON must be zero or positive"
[[ "$NUM_INFERENCE_STEPS" =~ ^[0-9]+$ ]] || die "NUM_INFERENCE_STEPS must be zero or positive"
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || die "MAX_STEPS must be zero or positive"
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TASKS must be -1 or positive"

# GPUS accepts either a comma- or space-separated list.  If omitted, use every
# device visible to the selected Python environment.  Each worker below gets a
# single visible device, so the evaluator always uses cuda:0 internally.
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
  [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || die "No CUDA GPU is visible to torch"
  GPU_TEXT=""
  for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
    GPU_TEXT+="${GPU_TEXT:+ }$gpu_index"
  done
fi
GPU_TEXT="${GPU_TEXT//,/ }"
read -r -a GPU_ARRAY <<< "$GPU_TEXT"
(( ${#GPU_ARRAY[@]} > 0 )) || die "No GPUs selected; set GPUS=0,1,..."
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || die "Duplicate GPU id: $gpu"
  SEEN_GPUS[$gpu]=1
done

WORLD_SIZE="${#GPU_ARRAY[@]}"
export FASTWAM_SOURCE_ROOT LEROBOT_ZIP MUJOCO_GL
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:$FASTWAM_SOURCE_ROOT/src:$FASTWAM_SOURCE_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$RESULT_ROOT" ]]; then
  [[ "$OVERWRITE" == "1" ]] || die "Result directory already exists: $RESULT_ROOT (set OVERWRITE=1 to reuse)"
else
  mkdir -p "$RESULT_ROOT"
fi

TASK_COUNT="$($PYTHON_BIN "$SCRIPT_DIR/eval_libero.py" \
  --model-path "$MODEL_PATH" --libero-root "$LIBERO_ROOT" \
  --output-dir "$RESULT_ROOT" --seed "${SEED_ARRAY[0]}" \
  --task-suites "${SUITE_ARRAY[@]}" --max-tasks "$MAX_TASKS" --dry-run \
  | tail -n 1 \
  | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["tasks_total"])'
)"
[[ "$TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || die "Could not determine LIBERO task count"
(( WORLD_SIZE <= TASK_COUNT )) || die "Selected GPUs ($WORLD_SIZE) exceed task count ($TASK_COUNT)"

info "FastWAM LIBERO three-seed evaluation"
info "Python: $PYTHON_BIN"
info "Model: $MODEL_PATH"
info "LIBERO: $LIBERO_ROOT"
info "LeRobot archive (optional compatibility path): $LEROBOT_ZIP"
info "Seeds: ${SEED_ARRAY[*]}"
info "Suites: ${SUITE_ARRAY[*]} ($TASK_COUNT tasks)"
info "Trials per task: $NUM_TRIALS_PER_TASK"
info "GPUs/workers: ${GPU_ARRAY[*]}"
info "Output: $RESULT_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  info "DRY_RUN=1; validation complete, no model workers were started"
  exit 0
fi

WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill -TERM "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

for seed in "${SEED_ARRAY[@]}"; do
  seed_dir="$RESULT_ROOT/seed$seed"
  if [[ -e "$seed_dir" && "$OVERWRITE" != "1" ]]; then
    die "Seed output already exists: $seed_dir"
  fi
  mkdir -p "$seed_dir/logs"
  info "Starting seed $seed"
  WORKER_PIDS=()
  worker_labels=()

  for ((rank = 0; rank < WORLD_SIZE; rank++)); do
    gpu="${GPU_ARRAY[$rank]}"
    log_file="$seed_dir/logs/worker_${rank}_gpu_${gpu}.log"
    worker_labels+=("rank=$rank gpu=$gpu")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero.py" \
        --model-path "$MODEL_PATH" \
        --libero-root "$LIBERO_ROOT" \
        --output-dir "$seed_dir" \
        --seed "$seed" \
        --rank "$rank" \
        --world-size "$WORLD_SIZE" \
        --gpu-id "$gpu" \
        --task-suites "${SUITE_ARRAY[@]}" \
        --num-trials "$NUM_TRIALS_PER_TASK" \
        --num-steps-wait "$NUM_STEPS_WAIT" \
        --replan-steps "$REPLAN_STEPS" \
        --action-horizon "$ACTION_HORIZON" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --max-steps "$MAX_STEPS" \
        --max-tasks "$MAX_TASKS" \
        ${NO_BINARIZE_GRIPPER:+--no-binarize-gripper}
    ) >"$log_file" 2>&1 &
    WORKER_PIDS+=("$!")
  done

  failed=0
  for index in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$index]}"; then
      info "Finished seed=$seed ${worker_labels[$index]}"
    else
      status=$?
      failed=1
      echo "[error] Failed seed=$seed ${worker_labels[$index]} (exit=$status); see $seed_dir/logs" >&2
    fi
  done
  if (( failed != 0 )); then
    cleanup
    die "Seed $seed failed"
  fi
  WORKER_PIDS=()

  "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero.py" \
    --model-path "$MODEL_PATH" --libero-root "$LIBERO_ROOT" \
    --output-dir "$seed_dir" --seed "$seed" \
    --aggregate-workers --expected-workers "$WORLD_SIZE" \
    --task-suites "${SUITE_ARRAY[@]}" >/dev/null
  seed_rate="$($PYTHON_BIN -c 'import json,sys; print("{:.6f}".format(json.load(open(sys.argv[1], encoding="utf-8"))["success_rate"]))' "$seed_dir/summary.json")"
  info "Seed $seed complete: success_rate=$seed_rate"
done

"$PYTHON_BIN" "$SCRIPT_DIR/eval_libero.py" \
  --model-path "$MODEL_PATH" --libero-root "$LIBERO_ROOT" \
  --output-dir "$RESULT_ROOT" --seed 0 \
  --aggregate-three-seed --output-root "$RESULT_ROOT" \
  --seeds "${SEED_ARRAY[@]}" >/dev/null

# Emit a compact per-subtask report in addition to the full aggregate.  The
# standalone summarizer also writes a Markdown table for quick inspection.
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_four_task_results.py" \
  --run-root "$RESULT_ROOT" --seeds "${SEED_ARRAY[@]}" >/dev/null

"$PYTHON_BIN" - "$RESULT_ROOT/three_seed_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"[ok] Three-seed arithmetic mean success rate: {summary['mean_success_rate']:.6f}")
print(f"[ok] Pooled success rate: {summary['pooled_success_rate']:.6f} "
      f"({summary['pooled_successes']}/{summary['pooled_episodes']})")
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    entry = summary.get("suites", {}).get(suite)
    if entry is None:
        continue
    print(f"[ok] {suite} pooled success rate: {entry['pooled_success_rate']:.6f} "
          f"({entry['pooled_successes']}/{entry['pooled_episodes']})")
print(f"[ok] Summary: {sys.argv[1]}")
print(f"[ok] Four-task summary: {sys.argv[1].replace('three_seed_summary.json', 'four_task_summary.json')}")
PY
