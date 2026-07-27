#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "$REPO_DIR/.." && pwd)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$(cd "$BENCHMARK_ROOT/.." && pwd)/LIBERO-plus}"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_plus}"

SUITE="${SUITE:-libero_10}"
MODEL_ROOT="${MODEL_ROOT:-}"  # set to the GR00T-N1.7 LIBERO checkpoint root
MODEL_PATH="${MODEL_PATH:-}"
COSMOS_PATH="${COSMOS_PATH:-}"  # set to a local Cosmos-Reason2-2B download
PATCH_COSMOS_PATH="${PATCH_COSMOS_PATH:-1}"
MODEL_NAME="${MODEL_NAME:-GR00T}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
BASE_PORT="${BASE_PORT:-7777}"
HOST="${HOST:-127.0.0.4}"
N_EPISODES="${N_EPISODES:-1}"
N_ENVS="${N_ENVS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
ONLY_FIRST_TASK="${ONLY_FIRST_TASK:-0}"
LIMIT_TASKS="${LIMIT_TASKS:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
OFFLINE="${OFFLINE:-1}"
RUN_NAME="${RUN_NAME:-${SUITE}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO_DIR/logs/libero_plus_eval}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SERVER_LOG_DIR="${LOG_DIR}/servers"
WORKER_LOG_DIR="${LOG_DIR}/workers"
RESULT_DIR="${LOG_DIR}/results"
RESULT_JSONL="${LOG_DIR}/results.jsonl"
SUMMARY_FILE="${LOG_DIR}/summary.md"
VIDEO_DIR="${VIDEO_DIR:-$REPO_DIR/data/libero_plus/videos/$RUN_NAME}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_DIR/data/libero_plus/config}"

case "$SUITE" in
  10|libero_10) SUITE="libero_10"; DEFAULT_MODEL_PATH="$MODEL_ROOT/libero_10" ;;
  goal|libero_goal) SUITE="libero_goal"; DEFAULT_MODEL_PATH="$MODEL_ROOT/libero_goal" ;;
  object|libero_object) SUITE="libero_object"; DEFAULT_MODEL_PATH="$MODEL_ROOT/libero_object" ;;
  spatial|libero_spatial) SUITE="libero_spatial"; DEFAULT_MODEL_PATH="$MODEL_ROOT/libero_spatial" ;;
  *) echo "[ERROR] Unknown SUITE=$SUITE. Use: libero_10, libero_goal, libero_object, libero_spatial" >&2; exit 1 ;;
esac
[[ -n "$MODEL_PATH" ]] || MODEL_PATH="$DEFAULT_MODEL_PATH"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$SERVER_LOG_DIR" "$WORKER_LOG_DIR" "$RESULT_DIR" "$VIDEO_DIR" "$LIBERO_CONFIG_PATH"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Cannot find conda.sh." >&2
  exit 1
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$LIBERO_PLUS_ROOT:$REPO_DIR:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset __GLX_VENDOR_LIBRARY_NAME || true
unset DISPLAY || true
unset XAUTHORITY || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

if [[ "$OFFLINE" == "1" ]]; then
  unset HF_ENDPOINT || true
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

LIBERO_INTERNAL_ROOT="$LIBERO_PLUS_ROOT/libero/libero"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PLUS_ROOT/libero/datasets
YAML

if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then echo "[ERROR] Missing model config.json: $MODEL_PATH/config.json" >&2; exit 1; fi
if [[ ! -f "$LIBERO_INTERNAL_ROOT/benchmark/task_classification.json" ]]; then echo "[ERROR] Missing LIBERO-plus task_classification.json" >&2; exit 1; fi

if [[ "$PATCH_COSMOS_PATH" == "1" && -d "$COSMOS_PATH" ]]; then
  python - <<PY
from pathlib import Path
model_dir = Path("${MODEL_PATH}")
old = "nvidia/Cosmos-Reason2-2B"
new = "${COSMOS_PATH}"
for path in model_dir.rglob("*"):
    if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".txt"}:
        text = path.read_text(errors="ignore")
        if old in text:
            print("patch:", path)
            path.write_text(text.replace(old, new))
PY
fi

wait_for_port() {
  local host="$1" port="$2" pid="$3" log_file="$4" max_tries="${5:-240}"
  for _ in $(seq 1 "$max_tries"); do
    if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1.0).close()
PY
    then
      echo "[INFO] Policy server ready at $host:$port"
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[ERROR] Policy server pid=$pid exited before ready." >&2
      tail -120 "$log_file" || true
      return 1
    fi
    sleep 1
  done
  echo "[ERROR] Policy server $host:$port not ready after ${max_tries}s." >&2
  tail -120 "$log_file" || true
  return 1
}

SERVER_PIDS=()
cleanup() {
  local code=$?
  if (( ${#SERVER_PIDS[@]} > 0 )); then
    echo "[INFO] Stopping policy servers: ${SERVER_PIDS[*]}"
    kill "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
    sleep 2
    kill -9 "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  exit "$code"
}
trap cleanup EXIT

read -r -a GPU_ARRAY <<< "$GPUS"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ARRAY[@]}}"
if (( WORLD_SIZE < 1 )); then echo "[ERROR] WORLD_SIZE must be >= 1" >&2; exit 1; fi
if (( ${#GPU_ARRAY[@]} < WORLD_SIZE )); then echo "[ERROR] GPUS has fewer entries than WORLD_SIZE=$WORLD_SIZE" >&2; exit 1; fi

cat <<INFO
========== GR00T LIBERO-plus suite eval ==========
REPO_DIR             = $REPO_DIR
CONDA_ENV            = $CONDA_ENV
python               = $(which python)
SUITE                = $SUITE
MODEL_PATH           = $MODEL_PATH
GPUS                 = $GPUS
WORLD_SIZE           = $WORLD_SIZE
BASE_PORT            = $BASE_PORT
N_EPISODES           = $N_EPISODES
N_ENVS               = $N_ENVS
N_ACTION_STEPS       = $N_ACTION_STEPS
MAX_EPISODE_STEPS    = $MAX_EPISODE_STEPS
ONLY_FIRST_TASK      = $ONLY_FIRST_TASK
LIMIT_TASKS          = $LIMIT_TASKS
SAVE_VIDEO           = $SAVE_VIDEO
LOG_DIR              = $LOG_DIR
==================================================
INFO

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$rank]}"
  port=$((BASE_PORT + rank))
  server_log="$SERVER_LOG_DIR/server_rank_${rank}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --port "$port" \
    > >(tee -a "$server_log") 2>&1 &
  SERVER_PIDS+=("$!")
  echo "[INFO] Started server rank=$rank gpu=$gpu port=$port pid=${SERVER_PIDS[$rank]}"
 done

for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  wait_for_port "$HOST" "$((BASE_PORT + rank))" "${SERVER_PIDS[$rank]}" "$SERVER_LOG_DIR/server_rank_${rank}.log" 240
 done

ONLY_FIRST_TASK_ARGS=()
[[ "$ONLY_FIRST_TASK" == "1" ]] && ONLY_FIRST_TASK_ARGS=(--only-first-task)
SAVE_VIDEO_ARGS=(--no-save-video)
case "${SAVE_VIDEO,,}" in true|1|yes|y) SAVE_VIDEO_ARGS=(--save-video) ;; esac

WORKER_PIDS=()
for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  gpu="${GPU_ARRAY[$rank]}"
  port=$((BASE_PORT + rank))
  worker_log="$WORKER_LOG_DIR/eval_rank_${rank}.log"
  result_file="$RESULT_DIR/rank_${rank}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_libero_plus_suite.py \
    --suite "$SUITE" \
    --classification-path "$LIBERO_INTERNAL_ROOT/benchmark/task_classification.json" \
    --result-jsonl "$result_file" \
    --summary-path "$LOG_DIR/summary_rank_${rank}.md" \
    --model-name "$MODEL_NAME" \
    --eval-rank "$rank" \
    --eval-world-size "$WORLD_SIZE" \
    --n-episodes "$N_EPISODES" \
    --n-envs "$N_ENVS" \
    --n-action-steps "$N_ACTION_STEPS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --policy-client-host "$HOST" \
    --policy-client-port "$port" \
    --video-dir "$VIDEO_DIR" \
    --limit-tasks "$LIMIT_TASKS" \
    --skip-summary \
    "${ONLY_FIRST_TASK_ARGS[@]}" \
    "${SAVE_VIDEO_ARGS[@]}" \
    2>&1 | tee "$worker_log" &
  WORKER_PIDS+=("$!")
  echo "[INFO] Started worker rank=$rank gpu=$gpu port=$port pid=${WORKER_PIDS[$rank]}"
 done

for pid in "${WORKER_PIDS[@]}"; do
  wait "$pid"
 done

: > "$RESULT_JSONL"
for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
  rank_file="$RESULT_DIR/rank_${rank}.jsonl"
  [[ -f "$rank_file" ]] && cat "$rank_file" >> "$RESULT_JSONL"
 done

python scripts/eval_libero_plus_suite.py \
  --suite "$SUITE" \
  --classification-path "$LIBERO_INTERNAL_ROOT/benchmark/task_classification.json" \
  --result-jsonl "$RESULT_JSONL" \
  --summary-path "$SUMMARY_FILE" \
  --model-name "$MODEL_NAME" \
  --summarize-only

echo "[INFO] Result JSONL: $RESULT_JSONL"
echo "[INFO] Summary: $SUMMARY_FILE"
