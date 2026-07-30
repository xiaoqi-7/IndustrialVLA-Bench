#!/usr/bin/env bash
# Evaluate FastWAM on LIBERO-Plus with one or more selected seeds.
# One model worker is launched per selected GPU.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-/root/envs/unifolm_libero/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"  # set to the FastWAM LIBERO checkpoint directory
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)/LIBERO-plus}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json}"
FASTWAM_SOURCE_ROOT="${FASTWAM_SOURCE_ROOT:-$BENCHMARK_ROOT/FastWAM}"
LEROBOT_ZIP="${LEROBOT_ZIP:-$BENCHMARK_ROOT/unifolm-vla/lerobot.zip}"

SEEDS_TEXT="${SEEDS:-1 7 42}"  # paper protocol: three run-level seeds
TASK_SUITES_TEXT="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-30}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TASKS="${MAX_TASKS:--1}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-$SCRIPT_DIR/results_plus}"
RESULT_ROOT="${RESULT_ROOT:-$RESULT_BASE/$RUN_NAME}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<'EOF'
Usage: bash FastWAM/run_fastwam_libero_plus.sh

Evaluates all 10,030 LIBERO-Plus tasks with the selected seeds.  Every
selected GPU runs one worker and receives a round-robin task shard.  The
official LIBERO-Plus default of one episode per task is used.  Each seed
writes summary.md with the seven robustness dimensions.  Existing completed
seed summaries under RESULT_ROOT are preserved and included in the run-root
summary together with newly completed seeds.  Run-level progress and current
rates are mirrored to total.log.

Examples:
  GPUS="0,1,2,3,4,5,6,7" bash FastWAM/run_fastwam_libero_plus.sh
  RESULT_ROOT=FastWAM/results_plus/20260717_122558 SEEDS="7 42" \
    GPUS="0,1,2,3,4,5,6,7" bash FastWAM/run_fastwam_libero_plus.sh
  DRY_RUN=1 GPUS="0,1" MAX_TASKS=20 bash FastWAM/run_fastwam_libero_plus.sh
  GPUS=0 MAX_TASKS=1 MAX_STEPS=5 NUM_STEPS_WAIT=0 \
    NUM_INFERENCE_STEPS=1 REPLAN_STEPS=2 bash FastWAM/run_fastwam_libero_plus.sh

Important variables: MODEL_PATH, LIBERO_PLUS_ROOT, PYTHON_BIN, GPUS, SEEDS,
TASK_SUITES, NUM_TRIALS_PER_TASK, MAX_TASKS, RESULT_BASE, RESULT_ROOT and
PROGRESS_LOG_EVERY.
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
[[ -f "$CLASSIFICATION_PATH" ]] || die "Missing LIBERO-Plus task classification: $CLASSIFICATION_PATH"
[[ -d "$LIBERO_PLUS_ROOT/libero/libero/bddl_files" ]] || die "Missing LIBERO-Plus BDDL directory"
[[ -d "$LIBERO_PLUS_ROOT/libero/libero/init_files" ]] || die "Missing LIBERO-Plus init directory"
[[ -d "$LIBERO_PLUS_ROOT/libero/libero/assets/new_objects" ]] || die "Missing extracted LIBERO-Plus assets/new_objects"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT//,/ }"
(( ${#SEED_ARRAY[@]} > 0 )) || die "At least one seed is required"
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
[[ "$FAIL_FAST" == "0" || "$FAIL_FAST" == "1" ]] || die "FAIL_FAST must be 0 or 1"
[[ "$PROGRESS_LOG_EVERY" =~ ^[1-9][0-9]*$ ]] || die "PROGRESS_LOG_EVERY must be positive"

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
export PYTHONPATH="$SCRIPT_DIR:$FASTWAM_SOURCE_ROOT/src:$FASTWAM_SOURCE_ROOT:$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

GRIPPER_ARGS=()
if [[ -n "${NO_BINARIZE_GRIPPER:-}" ]]; then
  GRIPPER_ARGS+=(--no-binarize-gripper)
fi
FAIL_FAST_ARGS=()
if [[ "$FAIL_FAST" == "1" ]]; then
  FAIL_FAST_ARGS+=(--fail-fast)
fi

if [[ -e "$RESULT_ROOT" ]]; then
  [[ -d "$RESULT_ROOT" ]] || die "Result path exists but is not a directory: $RESULT_ROOT"
else
  mkdir -p "$RESULT_ROOT"
fi

# Keep one run-level log in addition to the per-worker logs.  It captures the
# launcher lifecycle and the current seven-dimension metrics emitted after
# every completed seed.
TOTAL_LOG="$RESULT_ROOT/total.log"
FASTWAM_RUN_TOKEN="${FASTWAM_RUN_TOKEN:-${RUN_NAME}-$(date +%s)-$$}"
export FASTWAM_RUN_TOKEN PROGRESS_LOG_EVERY
touch "$TOTAL_LOG"
exec > >(tee -a "$TOTAL_LOG") 2>&1
info "Run started: $(date --iso-8601=seconds)"
info "Total log: $TOTAL_LOG"

TASK_COUNT="$($PYTHON_BIN "$SCRIPT_DIR/eval_libero_plus.py" \
  --model-path "$MODEL_PATH" --libero-plus-root "$LIBERO_PLUS_ROOT" \
  --classification-path "$CLASSIFICATION_PATH" \
  --output-dir "$RESULT_ROOT" --seed "${SEED_ARRAY[0]}" \
  --task-suites "${SUITE_ARRAY[@]}" --max-tasks "$MAX_TASKS" --dry-run \
  | tail -n 1 \
  | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["tasks_total"])'
)"
[[ "$TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || die "Could not determine LIBERO-Plus task count"
(( WORLD_SIZE <= TASK_COUNT )) || die "Selected GPUs ($WORLD_SIZE) exceed task count ($TASK_COUNT)"

info "FastWAM LIBERO-Plus selected-seed evaluation"
info "Python: $PYTHON_BIN"
info "Model: $MODEL_PATH"
info "LIBERO-Plus: $LIBERO_PLUS_ROOT"
info "Seeds: ${SEED_ARRAY[*]}"
info "Suites: ${SUITE_ARRAY[*]}"
info "Tasks: $TASK_COUNT"
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

COMPLETED_SEEDS=()
log_current_metrics() {
  local latest_seed="$1"
  shift
  "$PYTHON_BIN" - "$RESULT_ROOT" "$latest_seed" "$@" <<'PY'
import json
import sys
from pathlib import Path

columns = ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")
root = Path(sys.argv[1])
latest_seed = int(sys.argv[2])
seed_order = [int(seed) for seed in sys.argv[3:]]


def rate(entry):
    successes = int(entry.get("successes", 0))
    episodes = int(entry.get("episodes", 0))
    return successes / episodes if episodes else 0.0


results = []
for seed in seed_order:
    path = root / f"seed{seed}" / "summary.json"
    if path.is_file():
        results.append(json.loads(path.read_text(encoding="utf-8")))

latest = next((result for result in results if int(result["seed"]) == latest_seed), None)
if latest is not None:
    values = [100.0 * rate(latest.get("by_column", {}).get(column, {})) for column in columns]
    fields = " ".join(f"{column}={value:.2f}%" for column, value in zip(columns, values))
    print(
        f"[metrics] seed={latest['seed']} {fields} Total={100.0 * rate(latest):.2f}% "
        f"({latest['successes']}/{latest['episodes']}) task_errors={latest.get('errors', 0)}"
    )

if results:
    mean_values = []
    for column in columns:
        rates = [rate(result["by_column"][column]) for result in results if column in result.get("by_column", {})]
        mean_values.append(sum(rates) / len(rates) if rates else 0.0)
    total_rates = [rate(result) for result in results]
    fields = " ".join(
        f"{column}={100.0 * value:.2f}%" for column, value in zip(columns, mean_values)
    )
    completed = ",".join(str(result["seed"]) for result in results)
    print(
        f"[metrics] current_mean completed_seeds={completed} {fields} "
        f"Total={100.0 * sum(total_rates) / len(total_rates):.2f}%"
    )
PY
}

discover_completed_seeds() {
  "$PYTHON_BIN" - "$RESULT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = []
for path in root.glob("seed*/summary.json"):
    match = re.fullmatch(r"seed(-?\d+)", path.parent.name)
    if match is None:
        continue
    seed = int(match.group(1))
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if int(result.get("seed", seed)) != seed:
        continue
    seeds.append(seed)
for seed in sorted(set(seeds)):
    print(seed)
PY
}

# Include an already completed seed (for example seed 1 in a resumed run) in
# live means and in the final aggregate without evaluating it again.
mapfile -t COMPLETED_SEEDS < <(discover_completed_seeds)
if (( ${#COMPLETED_SEEDS[@]} > 0 )); then
  info "Existing completed seeds: ${COMPLETED_SEEDS[*]}"
fi

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
      "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_plus.py" \
        --model-path "$MODEL_PATH" \
        --libero-plus-root "$LIBERO_PLUS_ROOT" \
        --classification-path "$CLASSIFICATION_PATH" \
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
        "${GRIPPER_ARGS[@]}" \
        "${FAIL_FAST_ARGS[@]}"
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

  "$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_plus.py" \
    --model-path "$MODEL_PATH" --libero-plus-root "$LIBERO_PLUS_ROOT" \
    --output-dir "$seed_dir" --seed "$seed" \
    --aggregate-workers --expected-workers "$WORLD_SIZE" >/dev/null
  seed_rate="$($PYTHON_BIN -c 'import json,sys; print("{:.6f}".format(json.load(open(sys.argv[1], encoding="utf-8"))["success_rate"]))' "$seed_dir/summary.json")"
  info "Seed $seed complete: success_rate=$seed_rate"
  info "Seed $seed seven-dimension table: $seed_dir/summary.md"
  COMPLETED_SEEDS+=("$seed")
  log_current_metrics "$seed" "${COMPLETED_SEEDS[@]}"
done

mapfile -t SUMMARY_SEED_ARRAY < <(discover_completed_seeds)
(( ${#SUMMARY_SEED_ARRAY[@]} > 0 )) || die "No completed seed summaries found under $RESULT_ROOT"
info "Aggregating completed seeds: ${SUMMARY_SEED_ARRAY[*]}"
"$PYTHON_BIN" "$SCRIPT_DIR/eval_libero_plus.py" \
  --model-path "$MODEL_PATH" --libero-plus-root "$LIBERO_PLUS_ROOT" \
  --output-dir "$RESULT_ROOT" --seed 0 \
  --aggregate-three-seed --output-root "$RESULT_ROOT" \
  --seeds "${SUMMARY_SEED_ARRAY[@]}" >/dev/null

"$PYTHON_BIN" - "$RESULT_ROOT/three_seed_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    f"[ok] LIBERO-Plus {len(summary['seeds'])}-seed arithmetic mean success rate: "
    f"{summary['mean_success_rate']:.6f}"
)
for column in ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"):
    value = summary["by_column"].get(column, {}).get("mean_success_rate", 0.0)
    print(f"[ok] {column}: {value:.6f}")
print(f"[ok] Pooled success rate: {summary['pooled_success_rate']:.6f} "
      f"({summary['pooled_successes']}/{summary['pooled_episodes']})")
print(f"[ok] Summary JSON: {sys.argv[1]}")
root = Path(sys.argv[1]).parent
for seed in summary["seeds"]:
    print(f"[ok] Seed {seed} table: {root}/seed{seed}/summary.md")
print(f"[ok] Three-seed mean table: {root}/summary.md")
print(f"[ok] Compatibility table: {root}/three_seed_summary.md")
PY
