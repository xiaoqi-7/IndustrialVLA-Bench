#!/usr/bin/env bash

# Cosmos Policy LIBERO-plus evaluation on multiple GPUs.
#
# The four official LIBERO-plus suites are combined into one 10,030-task list
# and split as tasks[rank::world_size].  One independent model process is
# launched per selected GPU.  Seeds 1, 7, and 42 run strictly sequentially;
# within each seed, all selected GPUs run concurrently.
#
# Full evaluation:
#   GPUS=0,1,2,3,4,5,6,7 ./run_libero_plus_3seeds_multigpu.sh
#
# Short end-to-end smoke run (uses suite-specific horizons unless MAX_STEPS is
# set; this example intentionally truncates them):
#   GPUS=0,1 MAX_TASKS=8 MAX_STEPS=20 NUM_STEPS_WAIT=1 \
#     ./run_libero_plus_3seeds_multigpu.sh
#
# Reuse a previous partial run by passing the same RESULT_ROOT.  RESUME=1 is
# enabled by default and reads each rank's durable JSONL file.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/openpi/bin/python}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[info] $*"; }

MODEL_DIR="${MODEL_DIR:-/mnt/afs/raozf/models/Cosmos-Policy-LIBERO-Predict2-2B}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/afs/raozf/Cosmos-Predict2-2B-Video2World}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/mnt/afs/raozf/openpi/LIBERO-plus}"
LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"

CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt}"
DATASET_STATS="${DATASET_STATS:-$MODEL_DIR/libero_dataset_statistics.json}"
BASE_T5_CACHE="${BASE_T5_CACHE:-$MODEL_DIR/libero_t5_embeddings.pkl}"
T5_MODEL_DIR="${T5_MODEL_DIR:-$BASE_MODEL_DIR/text_encoder}"
T5_TOKENIZER_DIR="${T5_TOKENIZER_DIR:-$BASE_MODEL_DIR/tokenizer}"
# This compressed SQLite cache is persistent and shared read-only by workers.
T5_CACHE_PATH="${T5_CACHE_PATH:-$MODEL_DIR/libero_plus_t5_embeddings.sqlite3}"

SEEDS="${SEEDS:-1 7 42}"
SEEDS="${SEEDS//,/ }"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_SUITES="${TASK_SUITES//,/ }"
# Physical CUDA IDs.  If GPUS is omitted, select all cards that are currently
# mostly free.  Override GPUS explicitly when using a scheduler allocation.
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-70000}"
[[ "$MIN_FREE_GPU_MB" =~ ^[1-9][0-9]*$ ]] || die "MIN_FREE_GPU_MB must be positive"
GPUS="${GPUS:-}"
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk -v minimum="$MIN_FREE_GPU_MB" '$2 >= minimum {gsub(",", "", $1); print $1}' \
      | paste -sd, - || true)"
    [[ -n "$GPUS" ]] || die "No GPU has >=${MIN_FREE_GPU_MB} MiB free; set GPUS explicitly"
  else
    GPUS="0,1,2,3,4,5,6,7"
  fi
fi
GPU_LIST="${GPUS//,/ }"
read -r -a GPU_IDS <<< "$GPU_LIST"
WORLD_SIZE="${#GPU_IDS[@]}"

MAX_TASKS="${MAX_TASKS:--1}"
# -1 selects the official per-suite horizons: 220/280/300/520.
MAX_STEPS="${MAX_STEPS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-5}"
ENV_IMG_RES="${ENV_IMG_RES:-256}"
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
TASK_RETRIES="${TASK_RETRIES:-1}"
MAX_TASK_ERRORS="${MAX_TASK_ERRORS:-20}"
RESUME="${RESUME:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

PRECOMPUTE_T5="${PRECOMPUTE_T5:-1}"
PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-${GPU_IDS[0]:-}}"
T5_BATCH_SIZE="${T5_BATCH_SIZE:-16}"
T5_LOG_EVERY="${T5_LOG_EVERY:-100}"
T5_COMMIT_EVERY="${T5_COMMIT_EVERY:-100}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"

MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
MODEL_CONFIG="${MODEL_CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

RUN_NAME="${RUN_NAME:-cosmos_libero_plus_3seeds_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/libero_plus_3seeds/$RUN_NAME}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RESULT_ROOT/libero_config}"
# Preserve the existing /mnt/afs/raozf/cosmos-policy/summary.md (it currently
# contains another model's results).  Set PUBLISH_SUMMARY_PATH to that exact
# path if replacing it is intentional.
PUBLISH_SUMMARY_PATH="${PUBLISH_SUMMARY_PATH:-$SCRIPT_DIR/cosmos_policy_libero_plus_summary.md}"

[[ -x "$PYTHON_BIN" ]] || die "openpi Python not found: $PYTHON_BIN"
[[ -d "$LIBERO_PLUS_ROOT" ]] || die "LIBERO-plus root not found: $LIBERO_PLUS_ROOT"
[[ -d "$LIBERO_INTERNAL_ROOT/bddl_files" ]] || die "LIBERO-plus bddl_files not found"
[[ -d "$LIBERO_INTERNAL_ROOT/init_files" ]] || die "LIBERO-plus init_files not found"
[[ -d "$LIBERO_INTERNAL_ROOT/assets" ]] || die "LIBERO-plus assets not found"
[[ -f "$LIBERO_INTERNAL_ROOT/benchmark/task_classification.json" ]] \
  || die "LIBERO-plus task_classification.json not found"
[[ -f "$CHECKPOINT" ]] || die "Policy checkpoint not found: $CHECKPOINT"
[[ -f "$DATASET_STATS" ]] || die "Dataset stats not found: $DATASET_STATS"
[[ -f "$BASE_T5_CACHE" ]] || die "Base LIBERO T5 cache not found: $BASE_T5_CACHE"
[[ -f "$BASE_MODEL_DIR/model-480p-16fps.pt" ]] \
  || die "Base Cosmos checkpoint not found: $BASE_MODEL_DIR/model-480p-16fps.pt"
[[ -f "$BASE_MODEL_DIR/tokenizer/tokenizer.pth" ]] \
  || die "Base Cosmos tokenizer not found: $BASE_MODEL_DIR/tokenizer/tokenizer.pth"
[[ -f "$T5_MODEL_DIR/config.json" ]] || die "T5 config not found: $T5_MODEL_DIR/config.json"
[[ -f "$T5_TOKENIZER_DIR/spiece.model" ]] \
  || die "T5 tokenizer not found: $T5_TOKENIZER_DIR/spiece.model"

(( WORLD_SIZE > 0 )) || die "No GPUs selected; set GPUS=0,1,..."
[[ "$MAX_TASKS" =~ ^(-1|[1-9][0-9]*)$ ]] || die "MAX_TASKS must be -1 or positive"
[[ "$MAX_STEPS" =~ ^(-1|[1-9][0-9]*)$ ]] || die "MAX_STEPS must be -1 or positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || die "NUM_STEPS_WAIT must be non-negative"
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || die "CHUNK_SIZE must be positive"
[[ "$OPEN_LOOP_STEPS" =~ ^[1-9][0-9]*$ ]] || die "OPEN_LOOP_STEPS must be positive"
(( OPEN_LOOP_STEPS <= CHUNK_SIZE )) || die "OPEN_LOOP_STEPS must be <= CHUNK_SIZE"
[[ "$NUM_DENOISING_STEPS" =~ ^[1-9][0-9]*$ ]] || die "NUM_DENOISING_STEPS must be positive"
[[ "$ENVIRONMENT_SEED" =~ ^[0-9]+$ ]] || die "ENVIRONMENT_SEED must be non-negative"
[[ "$TASK_RETRIES" =~ ^[0-9]+$ ]] || die "TASK_RETRIES must be non-negative"
[[ "$MAX_TASK_ERRORS" =~ ^[0-9]+$ ]] || die "MAX_TASK_ERRORS must be non-negative"
[[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "POLL_INTERVAL must be positive"
[[ "$T5_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "T5_BATCH_SIZE must be positive"
[[ "$PRECOMPUTE_T5" == "0" || "$PRECOMPUTE_T5" == "1" ]] \
  || die "PRECOMPUTE_T5 must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || die "RESUME must be 0 or 1"
[[ "$PREFLIGHT_ONLY" == "0" || "$PREFLIGHT_ONLY" == "1" ]] \
  || die "PREFLIGHT_ONLY must be 0 or 1"

read -r -a SEED_IDS <<< "$SEEDS"
(( ${#SEED_IDS[@]} > 0 )) || die "SEEDS must not be empty"
declare -A SEEN_SEEDS=()
for seed in "${SEED_IDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
  [[ -z "${SEEN_SEEDS[$seed]+present}" ]] || die "Duplicate seed: $seed"
  SEEN_SEEDS[$seed]=1
done

declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu"
  [[ -z "${SEEN_GPUS[$gpu]+present}" ]] || die "Duplicate GPU id: $gpu"
  SEEN_GPUS[$gpu]=1
done
[[ "$PRECOMPUTE_GPU" =~ ^[0-9]+$ ]] || die "Invalid PRECOMPUTE_GPU: $PRECOMPUTE_GPU"

# Make the LIBERO-plus fork win over any regular LIBERO installation in the
# openpi environment.
export PYTHONPATH="$SCRIPT_DIR:$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$SCRIPT_DIR}"
export COSMOS_PREDICT2_BASE_MODEL_DIR="$BASE_MODEL_DIR"
export COSMOS_POLICY_CHECKPOINT="$CHECKPOINT"
export COSMOS_DATASET_STATS="$DATASET_STATS"
export LIBERO_T5_EMBEDDINGS="$BASE_T5_CACHE"
export LIBERO_PLUS_T5_CACHE="$T5_CACHE_PATH"
export COSMOS_T5_MODEL_DIR="$T5_MODEL_DIR"
export COSMOS_T5_TOKENIZER_DIR="$T5_TOKENIZER_DIR"
export MUJOCO_GL PYOPENGL_PLATFORM CUDA_DEVICE_ORDER
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "$RESULT_ROOT" "$LIBERO_CONFIG_PATH" "$RESULT_ROOT/datasets" "$(dirname "$T5_CACHE_PATH")"
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\nassets: %s\ndatasets: %s\n' \
  "$LIBERO_INTERNAL_ROOT" "$LIBERO_INTERNAL_ROOT/bddl_files" \
  "$LIBERO_INTERNAL_ROOT/init_files" "$LIBERO_INTERNAL_ROOT/assets" \
  "$RESULT_ROOT/datasets" > "$LIBERO_CONFIG_PATH/config.yaml"
export LIBERO_CONFIG_PATH

WORKER_MODULE="cosmos_policy.experiments.robot.libero.run_libero_plus_eval"
COMMON_MANIFEST_ARGS=(
  --task-suites "$TASK_SUITES"
  --max-tasks "$MAX_TASKS"
  --libero-config-path "$LIBERO_CONFIG_PATH"
)

info "Validating the official LIBERO-plus manifest ..."
"$PYTHON_BIN" -m "$WORKER_MODULE" --manifest-json "${COMMON_MANIFEST_ARGS[@]}" \
  > "$RESULT_ROOT/manifest.json"
TOTAL_TASKS="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_tasks"])' \
  "$RESULT_ROOT/manifest.json")"
UNIQUE_INSTRUCTIONS="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["unique_instructions"])' \
  "$RESULT_ROOT/manifest.json")"
[[ "$TOTAL_TASKS" =~ ^[1-9][0-9]*$ ]] || die "Invalid manifest task count: $TOTAL_TASKS"
if [[ "$MAX_TASKS" == "-1" && "$TASK_SUITES" == "libero_spatial libero_object libero_goal libero_10" ]]; then
  (( TOTAL_TASKS == 10030 )) || die "Full LIBERO-plus manifest must contain 10030 tasks, got $TOTAL_TASKS"
fi

# Dependency and physical GPU checks do not instantiate the policy model.
env -u CUDA_VISIBLE_DEVICES -u RANK -u LOCAL_RANK -u WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u LOCAL_WORLD_SIZE \
  GPU_LIST="${GPU_IDS[*]} $PRECOMPUTE_GPU" "$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys

for name in ("torch", "libero", "libero.libero.envs", "cosmos_policy", "transformers"):
    importlib.import_module(name)
import torch
print(f"[preflight] python={sys.executable}")
print(f"[preflight] torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is unavailable in the openpi environment")
selected = [int(item) for item in os.environ["GPU_LIST"].split()]
invalid = sorted({item for item in selected if item < 0 or item >= torch.cuda.device_count()})
if invalid:
    raise SystemExit(f"[error] Selected physical GPU id(s) unavailable: {invalid}")
print(f"[preflight] selected physical GPUs={selected}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  info "Current selected-GPU memory (MiB):"
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,memory.free \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' '{printf "  GPU %s: used=%s free=%s\n", $1, $2, $3}' || true
  done
fi

info "Python: $PYTHON_BIN (openpi environment)"
info "LIBERO-plus root: $LIBERO_PLUS_ROOT"
info "Tasks: $TOTAL_TASKS; unique policy prompts: $UNIQUE_INSTRUCTIONS"
info "Suites: $TASK_SUITES"
info "Seeds (sequential): $SEEDS"
info "GPUs (parallel within a seed): ${GPU_IDS[*]} (world_size=$WORLD_SIZE)"
info "Policy: $CHECKPOINT"
info "T5 cache: $T5_CACHE_PATH"
info "Results: $RESULT_ROOT"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  info "PREFLIGHT_ONLY=1: manifest, paths, imports, and GPU selection passed"
  exit 0
fi

if [[ "$PRECOMPUTE_T5" == "1" ]]; then
  info "Preparing/checking T5 cache on physical GPU $PRECOMPUTE_GPU ..."
  (
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$PRECOMPUTE_GPU" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" -m "$WORKER_MODULE" \
      --prepare-t5 \
      "${COMMON_MANIFEST_ARGS[@]}" \
      --base-t5-cache "$BASE_T5_CACHE" \
      --t5-cache "$T5_CACHE_PATH" \
      --t5-model-dir "$T5_MODEL_DIR" \
      --t5-tokenizer-dir "$T5_TOKENIZER_DIR" \
      --t5-device cuda:0 \
      --t5-batch-size "$T5_BATCH_SIZE" \
      --t5-log-every "$T5_LOG_EVERY" \
      --t5-commit-every "$T5_COMMIT_EVERY"
  )
else
  info "PRECOMPUTE_T5=0; validating the existing cache ..."
  "$PYTHON_BIN" -m "$WORKER_MODULE" \
    --check-t5-cache \
    "${COMMON_MANIFEST_ARGS[@]}" \
    --t5-cache "$T5_CACHE_PATH"
fi

declare -a ACTIVE_PIDS=()
cleanup() {
  if ((${#ACTIVE_PIDS[@]})); then
    info "Stopping ${#ACTIVE_PIDS[@]} active LIBERO-plus worker(s)"
    kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    wait "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    ACTIVE_PIDS=()
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

monitor_seed() {
  local seed="$1"
  local seed_dir="$RESULT_ROOT/seed${seed}"
  local marker
  marker="$(SEED="$seed" SEED_DIR="$seed_dir" EXPECTED="$TOTAL_TASKS" \
    EXPECTED_RANKS="$WORLD_SIZE" "$PYTHON_BIN" - <<'PY'
import glob
import json
import os
import time

category_to_short = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
seed = os.environ["SEED"]
seed_dir = os.environ["SEED_DIR"]
expected = int(os.environ["EXPECTED"])
expected_ranks = int(os.environ["EXPECTED_RANKS"])
completed = successes = total = task_errors = 0
statuses = []
started = None
category_stats = {name: [0, 0] for name in category_to_short}
for path in sorted(glob.glob(os.path.join(seed_dir, "progress", "rank*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    total += int(item.get("total", 0))
    completed += int(item.get("completed", 0))
    successes += int(item.get("successes", 0))
    task_errors += int(item.get("task_errors", 0))
    statuses.append(str(item.get("status", "unknown")))
    if item.get("started_at") is not None:
        value = float(item["started_at"])
        started = value if started is None else min(started, value)
    for category, values in item.get("category_stats", {}).items():
        if category in category_stats:
            category_stats[category][0] += int(values.get("successes", 0))
            category_stats[category][1] += int(values.get("completed", 0))

rate = successes / completed if completed else 0.0
elapsed = time.time() - started if started is not None else 0.0
done = (
    len(statuses) == expected_ranks
    and total == expected
    and completed == expected
    and all(status == "done" for status in statuses)
)
failed = any(status == "error" for status in statuses)
state = "DONE" if done else ("ERROR" if failed else "RUN")
parts = []
for category, short in category_to_short.items():
    success, count = category_stats[category]
    if count:
        parts.append(f"{short}={success / count * 100:.1f}%({success}/{count})")
category_line = ", ".join(parts) if parts else "categories=pending"
print(
    f"{state}|[seed={seed}] {completed}/{expected}, successes={successes}, "
    f"success_rate={rate * 100:.2f}%, task_errors={task_errors}, "
    f"elapsed={elapsed / 3600:.2f}h, ranks={len(statuses)}/{expected_ranks} | {category_line}"
)
PY
)"
  local state="${marker%%|*}"
  local line="${marker#*|}"
  echo "[$(date '+%F %T')] $line"
  MONITOR_STATE="$state"
}

run_seed() {
  local seed="$1"
  local seed_dir="$RESULT_ROOT/seed${seed}"
  mkdir -p "$seed_dir/progress"

  if [[ "$RESUME" == "1" && -f "$seed_dir/summary.json" ]] && \
    SEED="$seed" EXPECTED="$TOTAL_TASKS" SUMMARY="$seed_dir/summary.json" \
      "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import json, os
item = json.load(open(os.environ["SUMMARY"], encoding="utf-8"))
raise SystemExit(0 if int(item["seed"]) == int(os.environ["SEED"]) and int(item["total_tasks"]) == int(os.environ["EXPECTED"]) else 1)
PY
  then
    info "Seed $seed already has a complete summary; skipping workers"
    cat "$seed_dir/summary.md"
    return 0
  fi

  info "Starting seed=$seed with $WORLD_SIZE GPU workers"
  ACTIVE_PIDS=()
  local resume_arg="--resume"
  [[ "$RESUME" == "1" ]] || resume_arg="--no-resume"
  local rank
  for rank in "${!GPU_IDS[@]}"; do
    local gpu="${GPU_IDS[$rank]}"
    local stdout_log="$seed_dir/rank${rank}.stdout.log"
    info "  seed=$seed rank=$rank/$WORLD_SIZE -> physical GPU $gpu"
    (
      cd "$SCRIPT_DIR"
      env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u GROUP_RANK \
        -u GROUP_WORLD_SIZE -u LOCAL_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
        -u TORCHELASTIC_RUN_ID -u SLURM_PROCID -u SLURM_NTASKS \
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" -m "$WORKER_MODULE" \
        "${COMMON_MANIFEST_ARGS[@]}" \
        --seed "$seed" \
        --eval-rank "$rank" \
        --eval-world-size "$WORLD_SIZE" \
        --output-dir "$seed_dir" \
        --progress-file "$seed_dir/progress/rank${rank}.json" \
        --t5-cache "$T5_CACHE_PATH" \
        --config "$MODEL_CONFIG" \
        --ckpt-path "$CHECKPOINT" \
        --config-file "$CONFIG_FILE" \
        --dataset-stats "$DATASET_STATS" \
        --chunk-size "$CHUNK_SIZE" \
        --open-loop-steps "$OPEN_LOOP_STEPS" \
        --num-denoising-steps "$NUM_DENOISING_STEPS" \
        --max-steps "$MAX_STEPS" \
        --num-steps-wait "$NUM_STEPS_WAIT" \
        --env-img-res "$ENV_IMG_RES" \
        --environment-seed "$ENVIRONMENT_SEED" \
        --task-retries "$TASK_RETRIES" \
        --max-task-errors "$MAX_TASK_ERRORS" \
        "$resume_arg" \
        > "$stdout_log" 2>&1
    ) &
    ACTIVE_PIDS+=("$!")
  done

  local monitor_failed=0
  while true; do
    monitor_seed "$seed"
    if [[ "${MONITOR_STATE:-RUN}" == "ERROR" ]]; then
      echo "[error] seed=$seed reported a worker error; stopping remaining ranks" >&2
      kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
      monitor_failed=1
      break
    fi
    local all_finished=1
    local pid state
    for pid in "${ACTIVE_PIDS[@]}"; do
      if [[ -r "/proc/$pid/stat" ]]; then
        state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
        if [[ "$state" != "Z" ]]; then
          all_finished=0
          break
        fi
      fi
    done
    (( all_finished == 1 )) && break
    sleep "$POLL_INTERVAL"
  done

  local failed="$monitor_failed"
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
      echo "[error] seed=$seed worker pid=$pid failed; inspect $seed_dir/rank*.stdout.log" >&2
    fi
  done
  ACTIVE_PIDS=()
  monitor_seed "$seed"
  [[ "${MONITOR_STATE:-RUN}" == "DONE" ]] || failed=1
  (( failed == 0 )) || die "Seed $seed did not complete successfully"

  "$PYTHON_BIN" -m "$WORKER_MODULE" \
    --summarize-seed \
    --seed "$seed" \
    --output-dir "$seed_dir" \
    --expected-tasks "$TOTAL_TASKS" \
    --eval-world-size "$WORLD_SIZE"
}

# This loop is deliberately sequential.  The next seed starts only after all
# ranks of the current seed have exited and its summary.md is complete.
for seed in "${SEED_IDS[@]}"; do
  run_seed "$seed"
done

FINAL_SUMMARY_ARGS=(
  --summarize-final
  --seeds "$SEEDS"
  --result-root "$RESULT_ROOT"
  --expected-tasks "$TOTAL_TASKS"
)
if [[ -n "$PUBLISH_SUMMARY_PATH" ]]; then
  FINAL_SUMMARY_ARGS+=(--publish-summary "$PUBLISH_SUMMARY_PATH")
fi
"$PYTHON_BIN" -m "$WORKER_MODULE" "${FINAL_SUMMARY_ARGS[@]}"

info "LIBERO-plus evaluation complete"
for seed in "${SEED_IDS[@]}"; do
  info "Seed $seed summary: $RESULT_ROOT/seed${seed}/summary.md"
done
info "Three-seed summary: $RESULT_ROOT/summary.md"
[[ -n "$PUBLISH_SUMMARY_PATH" ]] && info "Published summary: $PUBLISH_SUMMARY_PATH"
