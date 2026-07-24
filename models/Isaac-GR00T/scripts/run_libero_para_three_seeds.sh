#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MULTI_GPU_LAUNCHER="$SCRIPT_DIR/eval_libero_para_multi_gpu.sh"
SUMMARIZER="$SCRIPT_DIR/summarize_libero_para_seeds.py"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_libero}"

if [[ "${1:-}" =~ ^(-h|--help|help)$ ]]; then
  cat <<'EOF'
Usage: run_libero_para_three_seeds.sh [SEED_1 SEED_2 SEED_3]

Examples:
  bash scripts/run_libero_para_three_seeds.sh 7 8 9
  SEEDS="7 42 123" GPUS="0 1 2 3" bash scripts/run_libero_para_three_seeds.sh

Runs three seeds sequentially. Within each seed, GPUS/WORLD_SIZE workers run in
parallel and every GPU loads the same libero_goal checkpoint. Outputs go under
logs/libero_para/gr00t/<experiment>/seedN. Set EXPERIMENT_NAME or OUTPUT_ROOT
to choose the experiment directory.
EOF
  exit 0
fi

if (( $# > 0 )); then
  SEED_ARRAY=("$@")
else
  read -r -a SEED_ARRAY <<< "${SEEDS:-7 8 9}"
fi

if (( ${#SEED_ARRAY[@]} != 3 )); then
  echo "[error] Exactly three seeds are required; got: ${SEED_ARRAY[*]}" >&2
  echo "Usage: $0 7 8 9   # or SEEDS='7 8 9' $0" >&2
  exit 2
fi

declare -A SEEN=()
for seed in "${SEED_ARRAY[@]}"; do
  [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "[error] Invalid seed: $seed" >&2; exit 2; }
  [[ -z "${SEEN[$seed]:-}" ]] || { echo "[error] Duplicate seed: $seed" >&2; exit 2; }
  SEEN[$seed]=1
done

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_libero_para_3seeds_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/logs/libero_para/gr00t/$EXPERIMENT_NAME}"
if [[ -e "$OUTPUT_ROOT/three_seed_summary.json" ]]; then
  echo "[error] Experiment output already exists: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

echo "[info] One checkpoint: libero_goal"
echo "[info] Three sequential seeds: ${SEED_ARRAY[*]}"
echo "[info] GPUs per seed: ${GPUS:-${CUDA_VISIBLE_DEVICES:-all torch-visible GPUs}}"
echo "[info] World size: ${WORLD_SIZE:-${NUM_WORKERS:-all listed GPUs}}"
echo "[info] Output root: $OUTPUT_ROOT"

for seed in "${SEED_ARRAY[@]}"; do
  echo
  echo "========== Starting seed $seed =========="
  OUTPUT_DIR="$OUTPUT_ROOT/seed$seed" \
  RUN_NAME="$EXPERIMENT_NAME" \
  CONDA_ENV="$CONDA_ENV" \
    bash "$MULTI_GPU_LAUNCHER" --seed "$seed"
  echo "========== Finished seed $seed =========="
done

"$CONDA_ENV/bin/python" "$SUMMARIZER" \
  --run-root "$OUTPUT_ROOT" \
  --seeds "${SEED_ARRAY[@]}"

echo "[ok] Three-seed experiment complete: $OUTPUT_ROOT"
