#!/usr/bin/env bash
# Evaluate FastWAM on all LIBERO-Para paraphrases with seeds 1, 7 and 42.
# One worker is launched per selected GPU; workers receive round-robin shards.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/root/envs/unifolm_libero/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"  # set to the FastWAM LIBERO checkpoint directory
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)/LIBERO-para}"
FASTWAM_SOURCE_ROOT="${FASTWAM_SOURCE_ROOT:-$BENCHMARK_ROOT/FastWAM}"
LEROBOT_ZIP="${LEROBOT_ZIP:-$BENCHMARK_ROOT/unifolm-vla/lerobot.zip}"

SEEDS_TEXT="${SEEDS:-1 7 42}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_TASKS="${MAX_TASKS:--1}"
RESOLUTION="${RESOLUTION:-360}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/libero_para_3seeds/$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<'EOF'
Usage: bash FastWAM/run_fastwam_libero_para.sh

Evaluates the 4,092 LIBERO-Para paraphrase tasks with seeds 1, 7 and 42,
using one worker per selected GPU.  By default each paraphrase is evaluated
once, matching the upstream LIBERO-Para evaluation scripts.

Examples:
  GPUS="0,1,2,3" bash FastWAM/run_fastwam_libero_para.sh
  GPUS=0 MAX_TASKS=2 NUM_TRIALS_PER_TASK=1 MAX_STEPS=5 \
    NUM_STEPS_WAIT=0 NUM_INFERENCE_STEPS=1 REPLAN_STEPS=2 \
    bash FastWAM/run_fastwam_libero_para.sh
  DRY_RUN=1 GPUS="0,1" MAX_TASKS=20 bash FastWAM/run_fastwam_libero_para.sh

Important variables: MODEL_PATH, LIBERO_PARA_ROOT, PYTHON_BIN, GPUS, SEEDS,
NUM_TRIALS_PER_TASK, MAX_TASKS, RESULT_ROOT and MUJOCO_GL.
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
[[ -d "$LIBERO_PARA_ROOT/libero/libero/bddl_files/libero_para" ]] || die "Missing LIBERO-Para BDDL directory"
[[ -d "$LIBERO_PARA_ROOT/libero/libero/bddl_files/libero_goal" ]] || die "Missing LIBERO-Goal BDDL directory"
[[ -d "$LIBERO_PARA_ROOT/libero/libero/init_files/libero_para" ]] || die "Missing LIBERO-Para init directory"
[[ -d "$LIBERO_PARA_ROOT/libero/libero/assets" ]] || die "Missing LIBERO-Para assets directory"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT//,/ }"
(( ${#SEED_ARRAY[@]} == 3 )) || die "Exactly three seeds are required; got: ${SEED_ARRAY[*]}"
declare -A SEEN_SEEDS=()
for seed in "${SEED_ARRAY[@]}"; do
  [[ "$seed" =~ ^-?[0-9]+$ ]] || die "Invalid seed: $seed"
  [[ -z "${SEEN_SEEDS[$seed]:-}" ]] || die "Duplicate seed: $seed"
  SEEN_SEEDS[$seed]=1
done

[[ "$NUM_TRIALS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || die "NUM_TRIALS_PER_TASK must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || die "NUM_STEPS_WAIT must be non-negative"
[[ "$REPLAN_STEPS" =~ ^[1-9][0-9]*$ ]] || die "REPLAN_STEPS must be positive"
[[ "$ACTION_HORIZON" =~ ^[0-9]+$ ]] || die "ACTION_HORIZON must be zero or positive"
[[ "$NUM_INFERENCE_STEPS" =~ ^[0-9]+$ ]] || die "NUM_INFERENCE_STEPS must be zero or positive"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be positive"
[[ "$MAX_TASKS" == "-1" || "$MAX_TASKS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TASKS must be -1 or positive"
[[ "$RESOLUTION" =~ ^[1-9][0-9]*$ ]] || die "RESOLUTION must be positive"

# GPUS accepts comma- or space-separated physical GPU IDs.  If omitted, use
# every device visible to the selected Python environment.
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
export PYTHONPATH="$SCRIPT_DIR:$FASTWAM_SOURCE_ROOT/src:$FASTWAM_SOURCE_ROOT:$LIBERO_PARA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$RESULT_ROOT" ]]; then
  [[ "$OVERWRITE" == "1" ]] || die "Result directory already exists: $RESULT_ROOT (set OVERWRITE=1 to reuse)"
else
  mkdir -p "$RESULT_ROOT"
fi

TASK_COUNT="$($PYTHON_BIN "$SCRIPT_DIR/eval_libero_para.py" \
  --model-path "$MODEL_PATH" --libero-para-root "$LIBERO_PARA_ROOT" \
  --output-dir "$RESULT_ROOT" --seed "${SEED_ARRAY[0]}" \
  --max-tasks "$MAX_TASKS" --dry-run \
  | tail -n 1 \
  | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["tasks_total"])'
)"
[[ "$TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || die "Could not determine LIBERO-Para task count"
(( WORLD_SIZE <= TASK_COUNT )) || die "Selected GPUs ($WORLD_SIZE) exceed task count ($TASK_COUNT)"

info "FastWAM LIBERO-Para three-seed evaluation"
info "Python: $PYTHON_BIN"
info "Model: $MODEL_PATH"
info "LIBERO-Para: $LIBERO_PARA_ROOT"
info "LeRobot archive (optional compatibility path): $LEROBOT_ZIP"
info "Seeds: ${SEED_ARRAY[*]}"
info "Tasks: $TASK_COUNT paraphrases"
info "Trials per paraphrase: $NUM_TRIALS_PER_TASK"
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
      "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_para.py" \
        --model-path "$MODEL_PATH" \
        --libero-para-root "$LIBERO_PARA_ROOT" \
        --output-dir "$seed_dir" \
        --seed "$seed" \
        --rank "$rank" \
        --world-size "$WORLD_SIZE" \
        --gpu-id "$gpu" \
        --num-trials "$NUM_TRIALS_PER_TASK" \
        --num-steps-wait "$NUM_STEPS_WAIT" \
        --replan-steps "$REPLAN_STEPS" \
        --action-horizon "$ACTION_HORIZON" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --max-steps "$MAX_STEPS" \
        --max-tasks "$MAX_TASKS" \
        --resolution "$RESOLUTION" \
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

  "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_para.py" \
    --model-path "$MODEL_PATH" --libero-para-root "$LIBERO_PARA_ROOT" \
    --output-dir "$seed_dir" --seed "$seed" \
    --aggregate-workers --expected-workers "$WORLD_SIZE" >/dev/null
  seed_rate="$($PYTHON_BIN -c 'import json,sys; print("{:.6f}".format(json.load(open(sys.argv[1], encoding="utf-8"))["success_rate"]))' "$seed_dir/summary.json")"
  info "Seed $seed complete: success_rate=$seed_rate"
done

"$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_para.py" \
  --model-path "$MODEL_PATH" --libero-para-root "$LIBERO_PARA_ROOT" \
  --output-dir "$RESULT_ROOT" --seed 0 \
  --aggregate-three-seed --output-root "$RESULT_ROOT" \
  --seeds "${SEED_ARRAY[@]}" >/dev/null

"$PYTHON_BIN" - "$RESULT_ROOT/three_seed_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"[ok] LIBERO-Para three-seed arithmetic mean success rate: {summary['mean_success_rate']:.6f}")
print(f"[ok] Pooled success rate: {summary['pooled_success_rate']:.6f} "
      f"({summary['pooled_successes']}/{summary['pooled_episodes']})")
print(f"[ok] Summary: {sys.argv[1]}")
PY
