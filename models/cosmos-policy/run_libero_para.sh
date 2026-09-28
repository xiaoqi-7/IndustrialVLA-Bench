#!/usr/bin/env bash

# Cosmos Policy LIBERO-Para multi-GPU evaluation.
#
# A worker is started on every selected GPU.  The 4,092 BDDL files are split
# deterministically as tasks[rank::world_size].  Progress JSON files are
# polled while workers run, so the global success rate is printed live.
# Seeds run sequentially to avoid keeping three model copies on every GPU.
#
# Examples:
#   GPUS=0,1,2,3 ./run_libero_para_3seeds_multigpu.sh
#   GPUS=4 MAX_TASKS=10 MAX_STEPS=40 NUM_STEPS_WAIT=1 \
#     ./run_libero_para_3seeds_multigpu.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/openpi/bin/python}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[info] $*"; }

MODEL_DIR="${MODEL_DIR:-/mnt/afs/raozf/models/Cosmos-Policy-LIBERO-Predict2-2B}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/afs/raozf/Cosmos-Predict2-2B-Video2World}"
PARA_ROOT="${PARA_ROOT:-/mnt/afs/raozf/openpi/LIBERO-Para}"
PARA_LIBERO_ROOT="$PARA_ROOT/libero/libero"
BDDL_DIR="${BDDL_DIR:-$PARA_LIBERO_ROOT/bddl_files/libero_para}"
INIT_DIR="${INIT_DIR:-$PARA_LIBERO_ROOT/init_files/libero_para}"
GOAL_BDDL_DIR="${GOAL_BDDL_DIR:-$PARA_LIBERO_ROOT/bddl_files/libero_goal}"

CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt}"
DATASET_STATS="${DATASET_STATS:-$MODEL_DIR/libero_dataset_statistics.json}"
BASE_T5_CACHE="${BASE_T5_CACHE:-$MODEL_DIR/libero_t5_embeddings.pkl}"
T5_MODEL_DIR="${T5_MODEL_DIR:-$BASE_MODEL_DIR/text_encoder}"
T5_TOKENIZER_DIR="${T5_TOKENIZER_DIR:-$BASE_MODEL_DIR/tokenizer}"
# Generated cache is persistent and does not overwrite the regular LIBERO
# cache.  Override T5_CACHE_PATH if MODEL_DIR is read-only.
T5_CACHE_PATH="${T5_CACHE_PATH:-$MODEL_DIR/libero_para_t5_embeddings.pkl}"

SEEDS="${SEEDS:-1 7 42}"
SEEDS="${SEEDS//,/ }"
# GPUS are physical CUDA IDs.  If omitted, select cards with at least 78 GiB
# free (useful when the regular LIBERO run is still occupying other cards).
# Set GPUS explicitly to force a particular multi-GPU allocation.
GPUS="${GPUS:-}"
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk '$2 >= 78000 {print $1}' | paste -sd, - || true)"
    [[ -n "$GPUS" ]] || die "No GPU with >=78 GiB free was auto-selected; set GPUS explicitly"
  else
    GPUS="0,1,2,3,4,5,6,7"
  fi
fi
GPU_LIST="${GPUS//,/ }"
read -r -a GPU_IDS <<< "$GPU_LIST"
WORLD_SIZE="${#GPU_IDS[@]}"

MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-5}"
ENV_IMG_RES="${ENV_IMG_RES:-256}"
T5_BATCH_SIZE="${T5_BATCH_SIZE:-4}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
PRECOMPUTE_T5="${PRECOMPUTE_T5:-1}"
PRECOMPUTE_GPU="${PRECOMPUTE_GPU:-}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

RUN_NAME="${RUN_NAME:-cosmos_libero_para_3seeds_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/libero_para_3seeds/$RUN_NAME}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RESULT_ROOT/libero_config}"
MODEL_CONFIG="${MODEL_CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"

[[ -x "$PYTHON_BIN" ]] || die "Python not found: $PYTHON_BIN"
[[ -d "$PARA_ROOT" ]] || die "LIBERO-Para root not found: $PARA_ROOT"
[[ -d "$BDDL_DIR" ]] || die "Para BDDL directory not found: $BDDL_DIR"
[[ -d "$INIT_DIR" ]] || die "Para init directory not found: $INIT_DIR"
[[ -d "$GOAL_BDDL_DIR" ]] || die "Goal BDDL directory not found: $GOAL_BDDL_DIR"
[[ -d "$PARA_LIBERO_ROOT/assets" ]] || die "Para assets directory not found: $PARA_LIBERO_ROOT/assets"
[[ -f "$CHECKPOINT" ]] || die "Policy checkpoint not found: $CHECKPOINT"
[[ -f "$DATASET_STATS" ]] || die "Dataset stats not found: $DATASET_STATS"
[[ -f "$BASE_MODEL_DIR/model-480p-16fps.pt" ]] || die "Base checkpoint not found: $BASE_MODEL_DIR/model-480p-16fps.pt"
[[ -f "$BASE_MODEL_DIR/tokenizer/tokenizer.pth" ]] || die "Base tokenizer not found: $BASE_MODEL_DIR/tokenizer/tokenizer.pth"
[[ -f "$T5_MODEL_DIR/config.json" ]] || die "T5 config not found: $T5_MODEL_DIR/config.json"
[[ -f "$T5_TOKENIZER_DIR/spiece.model" ]] || die "T5 tokenizer not found: $T5_TOKENIZER_DIR/spiece.model"

[[ "$MAX_TASKS" =~ ^(-1|[0-9]+)$ ]] || die "MAX_TASKS must be -1 or a non-negative integer"
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be positive"
[[ "$NUM_STEPS_WAIT" =~ ^[0-9]+$ ]] || die "NUM_STEPS_WAIT must be non-negative"
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || die "CHUNK_SIZE must be positive"
[[ "$OPEN_LOOP_STEPS" =~ ^[1-9][0-9]*$ ]] || die "OPEN_LOOP_STEPS must be positive"
(( OPEN_LOOP_STEPS <= CHUNK_SIZE )) || die "OPEN_LOOP_STEPS must be <= CHUNK_SIZE"
[[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "POLL_INTERVAL must be positive"
(( WORLD_SIZE > 0 )) || die "No GPUs selected; set GPUS=0,1,..."

read -r -a SEED_IDS <<< "$SEEDS"
(( ${#SEED_IDS[@]} > 0 )) || die "SEEDS must contain at least one integer"
declare -A SEEN_SEED_IDS=()
for seed in "${SEED_IDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
  [[ -z "${SEEN_SEED_IDS[$seed]+present}" ]] || die "Duplicate seed: $seed"
  SEEN_SEED_IDS[$seed]=1
done

declare -A SEEN_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu"
  [[ -z "${SEEN_GPU_IDS[$gpu]+present}" ]] || die "Duplicate GPU id: $gpu"
  SEEN_GPU_IDS[$gpu]=1
done

# If the caller did not choose a T5 card explicitly, prefer a selected GPU
# with at least 30 GiB currently free.  This avoids colliding with another
# evaluation during the one-time 11B T5 cache build.
if [[ -z "$PRECOMPUTE_GPU" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    for gpu in "${GPU_IDS[@]}"; do
      free_mb="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
      if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= 30000 )); then
        PRECOMPUTE_GPU="$gpu"
        break
      fi
    done
    [[ -n "$PRECOMPUTE_GPU" ]] || die "No selected GPU has >=30 GiB free for T5; set PRECOMPUTE_GPU explicitly"
  else
    PRECOMPUTE_GPU="${GPU_IDS[0]}"
  fi
fi

# The Para fork must precede any globally installed LIBERO package.
export PYTHONPATH="$SCRIPT_DIR:$PARA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$SCRIPT_DIR}"
export COSMOS_PREDICT2_BASE_MODEL_DIR="$BASE_MODEL_DIR"
export COSMOS_POLICY_CHECKPOINT="$CHECKPOINT"
export COSMOS_DATASET_STATS="$DATASET_STATS"
export LIBERO_T5_EMBEDDINGS="$BASE_T5_CACHE"
export COSMOS_T5_MODEL_DIR="$T5_MODEL_DIR"
export COSMOS_T5_TOKENIZER_DIR="$T5_TOKENIZER_DIR"
export MUJOCO_GL PYOPENGL_PLATFORM CUDA_DEVICE_ORDER
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "$RESULT_ROOT" "$LIBERO_CONFIG_PATH" "$RESULT_ROOT/datasets" "$(dirname "$T5_CACHE_PATH")"
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\nassets: %s\ndatasets: %s\n' \
  "$PARA_LIBERO_ROOT" "$PARA_LIBERO_ROOT/bddl_files" "$PARA_LIBERO_ROOT/init_files" \
  "$PARA_LIBERO_ROOT/assets" "$RESULT_ROOT/datasets" > "$LIBERO_CONFIG_PATH/config.yaml"
export LIBERO_CONFIG_PATH

TOTAL_TASKS="$(MAX_TASKS="$MAX_TASKS" BDDL_DIR="$BDDL_DIR" "$PYTHON_BIN" - <<'PY'
import glob, os
files = sorted(glob.glob(os.path.join(os.environ["BDDL_DIR"], "*.bddl")))
limit = int(os.environ["MAX_TASKS"])
if limit > 0:
    files = files[:limit]
print(len(files))
PY
)"
[[ "$TOTAL_TASKS" =~ ^[1-9][0-9]*$ ]] || die "No Para BDDL tasks found in $BDDL_DIR"

# Import checks and a CUDA visibility check without instantiating the model.
env -u CUDA_VISIBLE_DEVICES -u RANK -u LOCAL_RANK -u WORLD_SIZE -u GROUP_RANK \
  -u GROUP_WORLD_SIZE -u LOCAL_WORLD_SIZE GPU_LIST="$GPU_LIST" \
  PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" - <<'PY'
import importlib
import os
import torch
for name in ("libero", "libero.libero.envs", "cosmos_policy"):
    importlib.import_module(name)
print(f"[preflight] torch={torch.__version__}, cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
selected = [int(item) for item in os.environ["GPU_LIST"].split()]
invalid = [item for item in selected if item < 0 or item >= torch.cuda.device_count()]
if invalid:
    raise SystemExit(f"Selected GPU id(s) unavailable: {invalid}")
print(f"[preflight] selected GPUs={selected}")
PY

info "Para root: $PARA_ROOT"
info "Tasks: $TOTAL_TASKS BDDL files"
info "Seeds: $SEEDS"
info "GPUs: ${GPU_IDS[*]} (world_size=$WORLD_SIZE)"
info "Policy: $CHECKPOINT"
info "Base model: $BASE_MODEL_DIR"
info "T5 cache: $T5_CACHE_PATH"
info "Results: $RESULT_ROOT"

if [[ "$PRECOMPUTE_T5" == "1" ]]; then
  info "Preparing missing Para T5 embeddings on physical GPU $PRECOMPUTE_GPU ..."
  (
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$PRECOMPUTE_GPU" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" -m cosmos_policy.experiments.robot.libero.run_libero_para_eval \
      --prepare-t5 \
      --bddl-dir "$BDDL_DIR" \
      --base-t5-cache "$BASE_T5_CACHE" \
      --t5-cache "$T5_CACHE_PATH" \
      --t5-model-dir "$T5_MODEL_DIR" \
      --t5-tokenizer-dir "$T5_TOKENIZER_DIR" \
      --t5-device cuda:0 \
      --t5-batch-size "$T5_BATCH_SIZE" \
      --max-tasks "$MAX_TASKS"
  )
else
  info "PRECOMPUTE_T5=0; workers require a complete cache at $T5_CACHE_PATH"
fi
[[ -f "$T5_CACHE_PATH" ]] || die "T5 cache does not exist: $T5_CACHE_PATH"

declare -a ACTIVE_PIDS=()
cleanup() {
  if ((${#ACTIVE_PIDS[@]})); then
    info "Stopping ${#ACTIVE_PIDS[@]} active worker(s)"
    kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

monitor_seed() {
  local seed="$1"
  local seed_dir="$RESULT_ROOT/seed${seed}"
  local marker
  marker="$(SEED="$seed" SEED_DIR="$seed_dir" EXPECTED="$TOTAL_TASKS" "$PYTHON_BIN" - <<'PY'
import glob, json, os, time

seed = os.environ["SEED"]
seed_dir = os.environ["SEED_DIR"]
expected = int(os.environ["EXPECTED"])
completed = successes = total = 0
statuses = []
started = None
for path in sorted(glob.glob(os.path.join(seed_dir, "progress", "rank*.json"))):
    try:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    total += int(item.get("total", 0))
    completed += int(item.get("completed", 0))
    successes += int(item.get("successes", 0))
    statuses.append(str(item.get("status", "unknown")))
    if item.get("started_at") is not None:
        started = item["started_at"] if started is None else min(started, item["started_at"])
rate = successes / completed if completed else 0.0
elapsed = time.time() - started if started is not None else 0.0
eta = elapsed / completed * (expected - completed) if completed and expected > completed else 0.0
done = completed >= expected and total >= expected and len(statuses) > 0 and all(s == "done" for s in statuses)
failed = any(s == "error" for s in statuses)
state = "DONE" if done else ("ERROR" if failed else "RUN")
print(
    f"{state}|[seed={seed}] {completed}/{expected} tasks, successes={successes}, "
    f"success_rate={rate * 100:.2f}%, elapsed={elapsed / 3600:.2f}h, "
    f"ETA={eta / 3600:.2f}h, ranks={len(statuses)}"
)
PY
)"
  local state="${marker%%|*}"
  local line="${marker#*|}"
  echo "[$(date '+%F %T')] $line"
  MONITOR_STATE="$state"
}

aggregate_seed() {
  local seed="$1"
  local seed_dir="$RESULT_ROOT/seed${seed}"
  SEED="$seed" SEED_DIR="$seed_dir" EXPECTED="$TOTAL_TASKS" "$PYTHON_BIN" - <<'PY'
import json, os
from collections import defaultdict
from pathlib import Path

seed = os.environ["SEED"]
seed_dir = Path(os.environ["SEED_DIR"])
expected = int(os.environ["EXPECTED"])
rank_files = sorted(seed_dir.glob("rank*.json"))
if not rank_files:
    raise SystemExit(f"No rank result files under {seed_dir}")

records = []
for path in rank_files:
    with path.open("r", encoding="utf-8") as f:
        records.extend(json.load(f).get("records", []))
total = len(records)
successes = sum(bool(record.get("success", False)) for record in records)
if total != expected:
    raise SystemExit(f"Seed {seed}: got {total} records, expected {expected}")
records.sort(key=lambda record: int(record.get("task_id", 0)))

per_eval = defaultdict(lambda: {"total": 0, "successes": 0})
per_type = defaultdict(lambda: {"total": 0, "successes": 0})
by_eval = defaultdict(list)
for record in records:
    eval_id = int(record["eval_id"])
    key = f"eval{eval_id}"
    per_eval[key]["total"] += 1
    per_eval[key]["successes"] += int(bool(record.get("success", False)))
    type_key = str(record.get("paraphrase_type", "unknown"))
    per_type[type_key]["total"] += 1
    per_type[type_key]["successes"] += int(bool(record.get("success", False)))
    by_eval[eval_id].append(record)
for value in per_eval.values():
    value["success_rate"] = value["successes"] / value["total"] if value["total"] else 0.0
for value in per_type.values():
    value["success_rate"] = value["successes"] / value["total"] if value["total"] else 0.0

# Structured files are compatible with LIBERO-Para metrics/analyze_results.py.
for eval_id, eval_records in by_eval.items():
    (seed_dir / f"eval{eval_id}.json").write_text(
        json.dumps({"eval_id": eval_id, "original_instruction": None, "episodes": eval_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

rate = successes / total if total else 0.0
summary = {
    "seed": int(seed),
    "total_tasks": total,
    "successes": successes,
    "success_rate": rate,
    "success_rate_percent": rate * 100.0,
    "per_eval": dict(sorted(per_eval.items())),
    "per_type": dict(sorted(per_type.items())),
    "rank_files": [str(path) for path in rank_files],
}
(seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    f"===== Cosmos Policy LIBERO-Para seed {seed} =====",
    f"Tasks: {total}",
    f"Successes: {successes}",
    f"Success rate: {rate * 100:.2f}%",
    "",
    "Per base task:",
]
for key, value in sorted(per_eval.items(), key=lambda item: int(item[0][4:])):
    lines.append(f"  {key}: {value['success_rate'] * 100:.2f}% ({value['successes']}/{value['total']})")
lines.append("")
lines.append("Per paraphrase type:")
for key, value in sorted(per_type.items()):
    lines.append(f"  {key}: {value['success_rate'] * 100:.2f}% ({value['successes']}/{value['total']})")
(seed_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

run_seed() {
  local seed="$1"
  local seed_dir="$RESULT_ROOT/seed${seed}"
  mkdir -p "$seed_dir/progress"
  info "Starting seed=$seed with $WORLD_SIZE GPU worker(s)"
  ACTIVE_PIDS=()

  local rank
  for rank in "${!GPU_IDS[@]}"; do
    local gpu="${GPU_IDS[$rank]}"
    local stdout_log="$seed_dir/rank${rank}.stdout.log"
    info "  seed=$seed rank=$rank/$WORLD_SIZE -> GPU $gpu"
    (
      cd "$SCRIPT_DIR"
      env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u GROUP_RANK -u GROUP_WORLD_SIZE -u LOCAL_WORLD_SIZE \
        -u MASTER_ADDR -u MASTER_PORT -u TORCHELASTIC_RUN_ID \
        -u SLURM_PROCID -u SLURM_NTASKS \
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" -m cosmos_policy.experiments.robot.libero.run_libero_para_eval \
        --bddl-dir "$BDDL_DIR" \
        --init-dir "$INIT_DIR" \
        --goal-bddl-dir "$GOAL_BDDL_DIR" \
        --t5-cache "$T5_CACHE_PATH" \
        --seed "$seed" \
        --eval-rank "$rank" \
        --eval-world-size "$WORLD_SIZE" \
        --output-dir "$seed_dir" \
        --progress-file "$seed_dir/progress/rank${rank}.json" \
        --libero-config-path "$LIBERO_CONFIG_PATH" \
        --config "$MODEL_CONFIG" \
        --ckpt-path "$CHECKPOINT" \
        --config-file "$CONFIG_FILE" \
        --dataset-stats "$DATASET_STATS" \
        --max-tasks "$MAX_TASKS" \
        --chunk-size "$CHUNK_SIZE" \
        --open-loop-steps "$OPEN_LOOP_STEPS" \
        --num-denoising-steps "$NUM_DENOISING_STEPS" \
        --max-steps "$MAX_STEPS" \
        --num-steps-wait "$NUM_STEPS_WAIT" \
        --env-img-res "$ENV_IMG_RES" \
        > "$stdout_log" 2>&1
    ) &
    ACTIVE_PIDS+=("$!")
  done

  local all_finished=0
  while (( all_finished == 0 )); do
    monitor_seed "$seed"
    if [[ "${MONITOR_STATE:-RUN}" == "ERROR" ]]; then
      echo "[error] seed=$seed reported a worker error; stopping remaining ranks" >&2
      kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
    fi
    all_finished=1
    for pid in "${ACTIVE_PIDS[@]}"; do
      # A completed background subshell can remain as a zombie until wait is
      # called; kill -0 alone would keep the monitor loop alive forever.
      if [[ -r "/proc/$pid/stat" ]]; then
        state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
        if [[ "$state" != "Z" ]]; then
          all_finished=0
          break
        fi
      fi
    done
    (( all_finished == 1 )) || sleep "$POLL_INTERVAL"
  done

  local failed=0
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
      echo "[error] seed=$seed worker pid=$pid failed; inspect $seed_dir/*.stdout.log" >&2
    fi
  done
  monitor_seed "$seed"
  [[ "${MONITOR_STATE:-RUN}" == "DONE" ]] || failed=1
  (( failed == 0 )) || die "Seed $seed did not complete successfully"
  ACTIVE_PIDS=()
  aggregate_seed "$seed"
}

for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Invalid seed: $seed"
  run_seed "$seed"
done

# The requested final value is the arithmetic mean of the three per-seed
# success rates.  Pooled counts are printed as an additional diagnostic.
SEEDS="$SEEDS" RESULT_ROOT="$RESULT_ROOT" EXPECTED="$TOTAL_TASKS" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
seeds = os.environ["SEEDS"].split()
expected = int(os.environ["EXPECTED"])
per_seed = {}
for seed in seeds:
    path = root / f"seed{seed}" / "summary.json"
    if not path.is_file():
        raise SystemExit(f"Missing seed summary: {path}")
    item = json.loads(path.read_text(encoding="utf-8"))
    if int(item["total_tasks"]) != expected:
        raise SystemExit(f"Seed {seed} has {item['total_tasks']} tasks, expected {expected}")
    per_seed[seed] = item

mean_rate = sum(item["success_rate"] for item in per_seed.values()) / len(per_seed)
pooled_successes = sum(int(item["successes"]) for item in per_seed.values())
pooled_tasks = sum(int(item["total_tasks"]) for item in per_seed.values())
pooled_rate = pooled_successes / pooled_tasks if pooled_tasks else 0.0
summary = {
    "result_root": str(root),
    "seeds": [int(seed) for seed in seeds],
    "tasks_per_seed": expected,
    "per_seed": {
        seed: {
            "tasks": int(item["total_tasks"]),
            "successes": int(item["successes"]),
            "success_rate": float(item["success_rate"]),
            "success_rate_percent": float(item["success_rate_percent"]),
            "per_type": item.get("per_type", {}),
        }
        for seed, item in per_seed.items()
    },
    "three_seed_average": mean_rate,
    "three_seed_average_percent": mean_rate * 100.0,
    "pooled": {
        "tasks": pooled_tasks,
        "successes": pooled_successes,
        "success_rate": pooled_rate,
        "success_rate_percent": pooled_rate * 100.0,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    "===== Cosmos Policy LIBERO-Para final summary =====",
    f"Tasks per seed: {expected}",
    "",
]
for seed in seeds:
    item = per_seed[seed]
    lines.append(f"Seed {seed}: {item['success_rate_percent']:.2f}% ({item['successes']}/{item['total_tasks']})")
lines.extend([
    "",
    f"Three-seed average: {mean_rate * 100.0:.2f}%",
    f"Pooled result: {pooled_rate * 100.0:.2f}% ({pooled_successes}/{pooled_tasks})",
    "",
    f"Results: {root}",
])
(root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

info "LIBERO-Para evaluation complete"
info "Summary: $RESULT_ROOT/summary.txt"
