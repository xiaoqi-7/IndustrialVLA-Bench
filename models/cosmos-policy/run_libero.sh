#!/usr/bin/env bash

# Cosmos Policy LIBERO evaluation: seeds 1, 7, 42.
# The default job matrix is 3 seeds x 4 suites. After all jobs finish, a
# summary is printed per seed/suite and the three-seed mean is written to
# summary.txt and summary.json under RESULT_ROOT.
# Optional overrides:
#   GPUS=0,1,2,3 NUM_TRIALS_PER_TASK=1 bash run_libero_3seeds.sh
#   GPU=3 NUM_TRIALS_PER_TASK=1 bash run_libero_3seeds.sh  # single-card alias
#   NNODES=4 NODE_RANK=0 RUN_NAME=shared_run bash run_libero_3seeds.sh  # one invocation per node
#   AUTO_INSTALL_BASIC_DEPS=1 bash run_libero_3seeds.sh
#   SKIP_PREFLIGHT=1 bash run_libero_3seeds.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
# Use the current openpi conda environment by default.  Both values remain
# overridable via the environment for portability to another installation.
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/openpi/bin/python}"
MODEL_DIR="${MODEL_DIR:-}"  # set to the Cosmos-Policy-LIBERO-Predict2-2B checkpoint directory
# Local copy of the Cosmos Predict2 base model used by the policy config.  The
# policy checkpoint contains the fine-tuned network weights, but model setup
# still imports the base checkpoint and VAE paths from the experiment config.
# Point this at an existing local download on offline evaluation nodes.
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"  # set to a local Cosmos-Predict2-2B-Video2World download
LIBERO_PACKAGE_ROOT="${LIBERO_PACKAGE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)/LIBERO}"
LIBERO_INTERNAL_ROOT="$LIBERO_PACKAGE_ROOT/libero/libero"
SEEDS="${SEEDS:-1 7 42}"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
SEEDS="${SEEDS//,/ }"
TASK_SUITES="${TASK_SUITES//,/ }"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
# Multi-node sharding is scheduler-agnostic.  SLURM and torchrun's node
# variables are detected automatically; NNODES/NODE_RANK can always be set
# explicitly when launching one script process per node.  Nodes must share
# the model/LIBERO filesystem and use the same NNODES, SEEDS, TASK_SUITES,
# RUN_NAME, and RESULT_ROOT.
NNODES="${NNODES:-${NUM_NODES:-${SLURM_NNODES:-${GROUP_WORLD_SIZE:-${MLP_WORKER_NUM:-1}}}}}"
NODE_RANK="${NODE_RANK:-${NODE_INDEX:-${SLURM_NODEID:-${MLP_ROLE_INDEX:-${SLURM_PROCID:-${GROUP_RANK:-0}}}}}}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-${TORCHELASTIC_RUN_ID:-}}}"
RUN_NAME_INPUT="${RUN_NAME:-}"
DEFAULT_RUN_NAME="cosmos_libero_3seeds_${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/libero_3seeds/$RUN_NAME}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RESULT_ROOT/libero_config}"
RENDER_BACKEND="${RENDER_BACKEND:-osmesa}"
AUTO_INSTALL_BASIC_DEPS="${AUTO_INSTALL_BASIC_DEPS:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SUMMARY_WAIT_TIMEOUT="${SUMMARY_WAIT_TIMEOUT:-86400}"

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[info] $*"; }

[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be a non-negative integer"
(( NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
[[ "$SUMMARY_WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "SUMMARY_WAIT_TIMEOUT must be a positive integer"
if [[ -n "${LOCAL_WORLD_SIZE:-}" && "$LOCAL_WORLD_SIZE" != "1" ]]; then
  die "Launch one run_libero_3seeds.sh process per node (LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE)"
fi
if (( NNODES > 1 )) && [[ -z "$RUN_NAME_INPUT" && -z "$RUN_ID" ]]; then
  die "Multi-node runs require the same RUN_NAME=... on every node"
fi
# This is independent job sharding, not DDP/model-parallel execution.  Do not
# let a torchrun/scheduler launcher make the individual evaluation processes
# enter a second distributed process group.
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE LOCAL_WORLD_SIZE

# Each suite/seed is an independent evaluation.  By default, use all eight
# visible devices on the current node and run at most one evaluation process
# per device.  Set GPU=3 for backwards-compatible single-card execution, or
# GPUS=0,2,5 to select local devices.  MAX_PARALLEL_JOBS can further limit
# concurrency.  In a multi-node run, the complete seed/suite job list is
# sharded by NODE_RANK, so each job is executed exactly once across the nodes.
if [[ -n "${GPUS:-}" ]]; then
  GPU_LIST="$GPUS"
elif [[ -n "${GPU:-}" ]]; then
  GPU_LIST="$GPU"
else
  GPU_LIST="0,1,2,3,4,5,6,7"
fi
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPU_IDS <<< "$GPU_LIST"
[[ "${#GPU_IDS[@]}" -gt 0 ]] || die "No GPUs selected; set GPUS=0,1,..."
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu_id"
  [[ -z "${SEEN_GPU_IDS[$gpu_id]+present}" ]] || die "Duplicate GPU id: $gpu_id"
  SEEN_GPU_IDS[$gpu_id]=1
done
if [[ -n "${MAX_PARALLEL_JOBS:-}" ]]; then
  [[ "$MAX_PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]] || die "MAX_PARALLEL_JOBS must be a positive integer"
  GPU_IDS=("${GPU_IDS[@]:0:MAX_PARALLEL_JOBS}")
  [[ "${#GPU_IDS[@]}" -gt 0 ]] || die "MAX_PARALLEL_JOBS selected no GPUs"
fi
GPU_LIST="${GPU_IDS[*]}"

MODEL_CHECKPOINT="$MODEL_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DATASET_STATS="$MODEL_DIR/libero_dataset_statistics.json"
T5_EMBEDDINGS="$MODEL_DIR/libero_t5_embeddings.pkl"
DATASETS_ROOT="$RESULT_ROOT/libero_datasets"
BASE_MODEL_CHECKPOINT="$BASE_MODEL_DIR/model-480p-16fps.pt"
BASE_MODEL_TOKENIZER="$BASE_MODEL_DIR/tokenizer/tokenizer.pth"

[[ -x "$PYTHON_BIN" ]] || die "Python not found: $PYTHON_BIN"
[[ -f "$MODEL_DIR/config.json" ]] || die "Model config not found: $MODEL_DIR/config.json"
[[ -f "$MODEL_CHECKPOINT" ]] || die "Checkpoint not found: $MODEL_CHECKPOINT"
[[ -f "$DATASET_STATS" ]] || die "Dataset stats not found: $DATASET_STATS"
[[ -f "$T5_EMBEDDINGS" ]] || die "T5 embeddings not found: $T5_EMBEDDINGS"
[[ -f "$BASE_MODEL_CHECKPOINT" ]] || die "Base Cosmos Predict2 checkpoint not found: $BASE_MODEL_CHECKPOINT"
[[ -f "$BASE_MODEL_TOKENIZER" ]] || die "Base Cosmos Predict2 tokenizer not found: $BASE_MODEL_TOKENIZER"
[[ -d "$LIBERO_INTERNAL_ROOT/bddl_files" ]] || die "LIBERO bddl_files not found"
[[ -d "$LIBERO_INTERNAL_ROOT/init_files" ]] || die "LIBERO init_files not found"
[[ -d "$LIBERO_INTERNAL_ROOT/assets" ]] || die "LIBERO assets not found"

# Runtime paths for the copied MetaX-compatible environment.
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
[[ -d "$MACA_PATH/lib" ]] && export LD_LIBRARY_PATH="$MACA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -d "$MACA_PATH/mxgpu_llvm/lib" ]] && export LD_LIBRARY_PATH="$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH"

export PYTHONPATH="$SCRIPT_DIR:$LIBERO_PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$SCRIPT_DIR}"
# Let the experiment/checkpoint code resolve its built-in hf:// base-model
# references to the local download supplied above.
export COSMOS_PREDICT2_BASE_MODEL_DIR="$BASE_MODEL_DIR"
export MUJOCO_GL="$RENDER_BACKEND"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$RENDER_BACKEND}"
# All assets needed by this evaluation are present in the two local model
# directories.  Default to offline mode so importing unrelated experiment
# registrations cannot trigger network retries; callers may explicitly set
# HF_HUB_OFFLINE=0 and TRANSFORMERS_OFFLINE=0 to allow downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
# The checked-in LIBERO init-state files are pickled objects.  PyTorch 2.6+
# defaults torch.load to weights_only=True, so allow the trusted local files
# to load; callers can override this for stricter behavior.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export LIBERO_CONFIG_PATH TASK_SUITES NUM_TRIALS_PER_TASK GPU_LIST

mkdir -p "$RESULT_ROOT" "$LIBERO_CONFIG_PATH" "$DATASETS_ROOT"

# Force the regular 10-task LIBERO files; ~/.libero on this machine points to LIBERO-plus.
printf 'benchmark_root: %s\nbddl_files: %s\ninit_states: %s\nassets: %s\ndatasets: %s\n' \
  "$LIBERO_INTERNAL_ROOT" "$LIBERO_INTERNAL_ROOT/bddl_files" \
  "$LIBERO_INTERNAL_ROOT/init_files" "$LIBERO_INTERNAL_ROOT/assets" \
  "$DATASETS_ROOT" > "$LIBERO_CONFIG_PATH/config.yaml"

if [[ "$AUTO_INSTALL_BASIC_DEPS" == "1" ]] && ! "$PYTHON_BIN" -c 'import peft, megatron.core' >/dev/null 2>&1; then
  info "Installing peft and megatron-core..."
  "$PYTHON_BIN" -m pip install --disable-pip-version-check \
    "peft>=0.17.1" "megatron-core==0.14.0" "ipython"
fi

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  info "Running preflight checks..."
  "$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys

required = [
    "torch",
    "libero",
    "IPython",
    "megatron.core",
    "transformer_engine",
    "transformer_engine.pytorch",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    print("[error] Missing/incompatible modules:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("[hint] Use AUTO_INSTALL_BASIC_DEPS=1 for peft/megatron-core; "
          "transformer_engine needs a hardware-compatible build. NATTEN is not "
          "required by the default dense LIBERO configuration.", file=sys.stderr)
    raise SystemExit(2)

import torch
if not torch.cuda.is_available():
    raise SystemExit("[error] torch.cuda.is_available() is false")
print(f"[info] Python: {sys.executable}")
print(f"[info] PyTorch: {torch.__version__}")
print(f"[info] CUDA devices: {torch.cuda.device_count()}")
selected_gpus = [int(item) for item in os.environ["GPU_LIST"].split()]
invalid_gpus = [gpu for gpu in selected_gpus if gpu < 0 or gpu >= torch.cuda.device_count()]
if invalid_gpus:
    raise SystemExit(
        f"[error] Selected GPU id(s) {invalid_gpus} are unavailable; "
        f"torch reports {torch.cuda.device_count()} visible device(s)"
    )
print(f"[info] Selected GPUs: {selected_gpus}")

from libero.libero import benchmark
available = benchmark.get_benchmark_dict()
for suite in os.environ["TASK_SUITES"].split():
    if suite not in available:
        raise SystemExit(f"[error] Unknown LIBERO suite: {suite}")
    instance = available[suite]()
    if instance.n_tasks != 10:
        raise SystemExit(f"[error] {suite} has {instance.n_tasks} tasks; expected 10")
    states = instance.get_task_init_states(0)
    trials = int(os.environ["NUM_TRIALS_PER_TASK"])
    if len(states) < trials:
        raise SystemExit(f"[error] {suite} has only {len(states)} initial states")
    print(f"[info] {suite}: {instance.n_tasks} tasks, {len(states)} states/task")
PY
else
  info "SKIP_PREFLIGHT=1; dependency checks skipped"
fi

if [[ "${HF_HUB_OFFLINE:-0}" == "1" || "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
  info "HF offline mode enabled; base Cosmos Predict2 files must be cached"
else
  info "Base Cosmos Predict2 weights/tokenizer may download on first run"
fi

info "Model: $MODEL_CHECKPOINT"
info "Base model: $BASE_MODEL_DIR"
info "Base checkpoint: $BASE_MODEL_CHECKPOINT"
info "NATTEN: disabled (dense minimal_a2a attention)"
info "Seeds: $SEEDS"
info "Suites: $TASK_SUITES"
info "Trials/task: $NUM_TRIALS_PER_TASK"
info "GPUs: CUDA_VISIBLE_DEVICES per job = $GPU_LIST"
info "Parallel jobs: ${#GPU_IDS[@]}"
info "Node shard: rank=$NODE_RANK/$NNODES"
info "Results: $RESULT_ROOT"

run_evaluation() {
  local seed="$1"
  local suite="$2"
  local gpu="$3"
  local run_dir="$RESULT_ROOT/$suite/seed$seed"
  local log_dir="$run_dir/logs"
  local status

  mkdir -p "$run_dir" "$log_dir"
  (
    cd "$run_dir" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
      -m cosmos_policy.experiments.robot.libero.run_libero_eval \
      --config cosmos_predict2_2b_480p_libero__inference_only \
      --ckpt_path "$MODEL_CHECKPOINT" \
      --config_file cosmos_policy/config/config.py \
      --use_third_person_image True \
      --use_wrist_image True \
      --use_proprio True \
      --normalize_proprio True \
      --unnormalize_actions True \
      --dataset_stats_path "$DATASET_STATS" \
      --t5_text_embeddings_path "$T5_EMBEDDINGS" \
      --trained_with_image_aug True \
      --chunk_size 16 \
      --num_open_loop_steps 16 \
      --task_suite_name "$suite" \
      --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
      --local_log_dir "$log_dir" \
      --randomize_seed False \
      --data_collection False \
      --seed "$seed" \
      --use_variance_scale False \
      --deterministic True \
      --run_id_note "seed$seed-$suite" \
      --ar_future_prediction False \
      --ar_value_prediction False \
      --use_jpeg_compression True \
      --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 \
      --num_denoising_steps_value 1 \
      --use_wandb False
  ) 2>&1 | tee "$run_dir/console.log"
  status=${PIPESTATUS[0]}
  printf '%s\n' "$status" > "$run_dir/exit_status"
  return "$status"
}

# Build the independent jobs and keep only this node's deterministic shard.
# Jobs are launched in batches, which keeps the implementation portable to
# Bash versions without wait -n while still using all selected local cards.
JOB_SEEDS=()
JOB_SUITES=()
global_job_index=0
for seed in $SEEDS; do
  for suite in $TASK_SUITES; do
    if (( global_job_index % NNODES == NODE_RANK )); then
      JOB_SEEDS+=("$seed")
      JOB_SUITES+=("$suite")
    fi
    global_job_index=$((global_job_index + 1))
  done
done

total_jobs="${#JOB_SEEDS[@]}"
info "Assigned jobs on node $NODE_RANK: $total_jobs/$global_job_index"
if (( total_jobs == 0 )); then
  info "No jobs assigned to this node; exiting successfully"
  exit 0
fi
job_index=0
batch_index=0
while (( job_index < total_jobs )); do
  pids=()
  labels=()
  gpu_index=0

  while (( gpu_index < ${#GPU_IDS[@]} && job_index < total_jobs )); do
    seed="${JOB_SEEDS[$job_index]}"
    suite="${JOB_SUITES[$job_index]}"
    gpu="${GPU_IDS[$gpu_index]}"
    info "Launching suite=$suite seed=$seed on GPU=$gpu (batch=$batch_index)"
    run_evaluation "$seed" "$suite" "$gpu" &
    pids+=("$!")
    labels+=("suite=$suite seed=$seed GPU=$gpu")
    job_index=$((job_index + 1))
    gpu_index=$((gpu_index + 1))
  done

  batch_failed=0
  for pid_index in "${!pids[@]}"; do
    if wait "${pids[$pid_index]}"; then
      info "Finished ${labels[$pid_index]}"
    else
      status=$?
      batch_failed=1
      echo "[error] Failed ${labels[$pid_index]} (exit=$status); see the corresponding console.log" >&2
    fi
  done
  (( batch_failed == 0 )) || die "One or more evaluations failed in batch=$batch_index"
  batch_index=$((batch_index + 1))
done

# Parse the final counters emitted by each evaluation process and print one
# consolidated report. Keeping this aggregation here (after all child
# processes have been waited on) avoids races between concurrently running
# suite/seed jobs and makes the result reproducible on multi-node launches.
summarize_results() {
  local summary_json="$RESULT_ROOT/summary.json"
  local summary_txt="$RESULT_ROOT/summary.txt"

  RESULT_ROOT="$RESULT_ROOT" \
  SEEDS="$SEEDS" \
  TASK_SUITES="$TASK_SUITES" \
  NUM_TRIALS_PER_TASK="$NUM_TRIALS_PER_TASK" \
  SUMMARY_JSON="$summary_json" \
  SUMMARY_TXT="$summary_txt" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
import re
import sys
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
seeds = os.environ["SEEDS"].split()
suites = os.environ["TASK_SUITES"].split()
num_trials = int(os.environ["NUM_TRIALS_PER_TASK"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_txt = Path(os.environ["SUMMARY_TXT"])

rate_re = re.compile(r"Overall success rate:\s*([0-9]+(?:\.[0-9]+)?)")
episodes_re = re.compile(r"Total episodes:\s*(\d+)")
successes_re = re.compile(r"Total successes:\s*(\d+)")


def find_log(suite: str, seed: str) -> Path | None:
    run_dir = result_root / suite / f"seed{seed}"
    console_log = run_dir / "console.log"
    if console_log.is_file():
        return console_log
    # Keep the summary useful if a caller invokes the evaluator directly and
    # only the evaluator's normal local .txt log is present.
    candidates = sorted(run_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


results = {}
missing = []
for seed in seeds:
    results[seed] = {}
    for suite in suites:
        log_path = find_log(suite, seed)
        if log_path is None:
            missing.append(f"{suite}/seed{seed}")
            continue
        text = log_path.read_text(errors="replace")
        rate_matches = rate_re.findall(text)
        episode_matches = episodes_re.findall(text)
        success_matches = successes_re.findall(text)
        if not rate_matches or not episode_matches or not success_matches:
            missing.append(f"{suite}/seed{seed} (incomplete log: {log_path})")
            continue
        episodes = int(episode_matches[-1])
        successes = int(success_matches[-1])
        # Counts are authoritative; the logged floating-point rate is kept for
        # diagnostics and should agree up to formatting/rounding.
        rate = successes / episodes if episodes else 0.0
        logged_rate = float(rate_matches[-1])
        results[seed][suite] = {
            "episodes": episodes,
            "successes": successes,
            "rate": rate,
            "rate_percent": rate * 100.0,
            "logged_rate": logged_rate,
            "log": str(log_path),
        }

if missing:
    print("[error] Cannot build LIBERO summary; missing/incomplete jobs:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(2)

seed_averages = {}
for seed in seeds:
    seed_averages[seed] = (
        sum(results[seed][suite]["rate"] for suite in suites) / len(suites) if suites else 0.0
    )

suite_averages = {}
for suite in suites:
    suite_averages[suite] = (
        sum(results[seed][suite]["rate"] for seed in seeds) / len(seeds) if seeds else 0.0
    )

three_seed_average = sum(seed_averages.values()) / len(seed_averages) if seed_averages else 0.0
total_episodes = sum(results[seed][suite]["episodes"] for seed in seeds for suite in suites)
total_successes = sum(results[seed][suite]["successes"] for seed in seeds for suite in suites)
pooled_rate = total_successes / total_episodes if total_episodes else 0.0

summary = {
    "result_root": str(result_root),
    "seeds": seeds,
    "task_suites": suites,
    "num_trials_per_task": num_trials,
    "results": results,
    "seed_averages": {
        seed: {"rate": rate, "rate_percent": rate * 100.0}
        for seed, rate in seed_averages.items()
    },
    "suite_averages": {
        suite: {"rate": rate, "rate_percent": rate * 100.0}
        for suite, rate in suite_averages.items()
    },
    "three_seed_average": {
        "rate": three_seed_average,
        "rate_percent": three_seed_average * 100.0,
    },
    "pooled_total": {
        "episodes": total_episodes,
        "successes": total_successes,
        "rate": pooled_rate,
        "rate_percent": pooled_rate * 100.0,
    },
}

summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

lines = []
lines.append("===== Cosmos Policy LIBERO summary =====")
lines.append(f"Suites: {', '.join(suites)}")
lines.append(f"Seeds: {', '.join(seeds)}")
lines.append(f"Trials per task: {num_trials}")
lines.append("")
for seed in seeds:
    lines.append(f"Seed {seed}:")
    for suite in suites:
        item = results[seed][suite]
        lines.append(
            f"  {suite}: {item['rate_percent']:.2f}% "
            f"({item['successes']}/{item['episodes']})"
        )
    lines.append(f"  Seed average ({len(suites)} suites): {seed_averages[seed] * 100.0:.2f}%")
    lines.append("")

lines.append("Three-seed average (mean of seed averages): " f"{three_seed_average * 100.0:.2f}%")
lines.append(f"Pooled result (all jobs): {pooled_rate * 100.0:.2f}% ({total_successes}/{total_episodes})")
lines.append("")
lines.append("Per-suite average across seeds:")
for suite in suites:
    lines.append(f"  {suite}: {suite_averages[suite] * 100.0:.2f}%")
lines.append("")
lines.append(f"JSON: {summary_json}")
lines.append(f"Text: {summary_txt}")

summary_txt.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

if (( NNODES == 1 )); then
  summarize_results
elif (( NODE_RANK == 0 )); then
  # Other nodes write their console.log/exit_status files to the shared
  # RESULT_ROOT. Wait for every assigned job before aggregating on rank 0.
  info "Waiting for all $global_job_index node-sharded jobs before building the summary..."
  wait_start="$(date +%s)"
  while :; do
    missing_jobs=0
    failed_jobs=0
    for seed in $SEEDS; do
      for suite in $TASK_SUITES; do
        job_dir="$RESULT_ROOT/$suite/seed$seed"
        if [[ ! -f "$job_dir/exit_status" ]]; then
          missing_jobs=$((missing_jobs + 1))
        elif [[ "$(<"$job_dir/exit_status")" != "0" ]]; then
          failed_jobs=$((failed_jobs + 1))
        fi
      done
    done
    (( failed_jobs == 0 )) || die "A node-sharded evaluation failed; inspect its exit_status/console.log"
    (( missing_jobs == 0 )) && break
    now="$(date +%s)"
    if (( now - wait_start >= SUMMARY_WAIT_TIMEOUT )); then
      die "Timed out waiting for $missing_jobs node-sharded jobs (SUMMARY_WAIT_TIMEOUT=$SUMMARY_WAIT_TIMEOUT s)"
    fi
    if (( (now - wait_start) % 60 < 5 )); then
      info "Still waiting for $missing_jobs job(s); elapsed=$((now - wait_start))s"
    fi
    sleep 5
  done
  summarize_results
else
  info "Node rank $NODE_RANK completed its shard; rank 0 will print the shared summary"
fi

info "All Cosmos Policy LIBERO evaluations completed"
info "Results: $RESULT_ROOT"
