#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MULTI_GPU_LAUNCHER="$SCRIPT_DIR/eval_libero_para_multi_gpu.sh"
SUMMARIZER="$SCRIPT_DIR/summarize_libero_para_seeds.py"

if [[ "${1:-}" =~ ^(-h|--help|help)$ ]]; then
  cat <<'EOF'
Usage: run_libero_para_three_seeds.sh [SEED_1 SEED_2 SEED_3]

Examples:
  GPUS="0 1 2 3 4 5 6 7" bash scripts/run_libero_para_three_seeds.sh 7 8 9
  SEEDS="7 42 123" GPUS="0 1 2 3" bash scripts/run_libero_para_three_seeds.sh

Runs exactly three seeds sequentially. Within each seed, GPUS/WORLD_SIZE
workers run in parallel and every GPU loads the same converted pi05_libero
PyTorch checkpoint. Outputs are written under
logs/libero_para/openpi/<experiment>/seedN, never under results/.
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

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x /root/miniconda3/envs/openpi/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/openpi/bin/python
elif [[ -x /root/envs/openpi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_plus/bin/python
elif [[ -x /root/envs/openpi-plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi-plus/bin/python
elif [[ -x /root/envs/openpi_libero/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_libero/bin/python
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  echo "[error] Cannot find an OpenPI Python environment. Set PYTHON=/path/to/python." >&2
  exit 1
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-openpi_para_3seeds_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OPENPI_ROOT/logs/libero_para/openpi/$EXPERIMENT_NAME}"

if [[ -e "$OUTPUT_ROOT/three_seed_summary.json" ]]; then
  echo "[error] Experiment output already exists: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

echo "[info] Model: OpenPI pi0.5 LIBERO PyTorch checkpoint"
echo "[info] Three sequential seeds: ${SEED_ARRAY[*]}"
echo "[info] GPUs per seed: ${GPUS:-${CUDA_VISIBLE_DEVICES:-all torch-visible GPUs}}"
echo "[info] World size: ${WORLD_SIZE:-${NUM_WORKERS:-all listed GPUs}}"
echo "[info] Output root: $OUTPUT_ROOT"

for seed in "${SEED_ARRAY[@]}"; do
  echo
  echo "========== Starting seed $seed =========="
  OUTPUT_DIR="$OUTPUT_ROOT/seed$seed" \
  RUN_NAME="$EXPERIMENT_NAME" \
    bash "$MULTI_GPU_LAUNCHER" --seed "$seed"
  echo "========== Finished seed $seed =========="
done

"$PYTHON_BIN" "$SUMMARIZER" \
  --run-root "$OUTPUT_ROOT" \
  --seeds "${SEED_ARRAY[@]}"

echo "[ok] OpenPI PyTorch three-seed experiment complete: $OUTPUT_ROOT"
