#!/usr/bin/env bash
set -euo pipefail

# Evaluate OpenPI on LIBERO-Para with the local PyTorch checkpoint.
# Runs the requested seeds sequentially and writes LIBERO-Para structured logs:
#   RESULT_ROOT/seed1/eval0.json ... summary.json
#   RESULT_ROOT/seed7/eval0.json ... summary.json
#   RESULT_ROOT/seed42/eval0.json ... summary.json
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$OPENPI_ROOT/LIBERO-Para}"
TORCH_CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR:-}"  # set to the converted pi05_libero PyTorch checkpoint directory
MODEL_NAME="${MODEL_NAME:-OpenPI-Torch}"

SEEDS="${SEEDS:-1 7 42}"
GPUS="${GPUS:-${RANKS:-1 2 6 7}}"
GPUS="${GPUS//,/ }"
read -r -a GPU_ARRAY <<< "$GPUS"
WORLD_SIZE="${WORLD_SIZE:-${#GPU_ARRAY[@]}}"
if (( WORLD_SIZE < 1 )); then
  echo "[error] WORLD_SIZE must be >= 1." >&2
  exit 1
fi
if (( ${#GPU_ARRAY[@]} < WORLD_SIZE )); then
  echo "[error] GPUS='$GPUS' has fewer entries than WORLD_SIZE=$WORLD_SIZE." >&2
  exit 1
fi

PORT="${PORT:-6666}"
HOST="${HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_WAIT_TIMEOUT_S="${SERVER_WAIT_TIMEOUT_S:-3600}"
SERVER_WAIT_POLL_S="${SERVER_WAIT_POLL_S:-5}"

RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_TASKS="${MAX_TASKS:--1}"

RUN_NAME="${RUN_NAME:-torch_para_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$OPENPI_ROOT/logs/libero_para_eval/$RUN_NAME}"
SERVER_LOG_ROOT="${SERVER_LOG_ROOT:-$OPENPI_ROOT/logs/openpi_para_servers_torch/$RUN_NAME}"
RESULT_ROOT="${RESULT_ROOT:-$OPENPI_ROOT/logs/libero_para_eval/$RUN_NAME/results}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$OPENPI_ROOT/data/libero_para/config}"

LIBERO_INTERNAL_ROOT="$LIBERO_PARA_ROOT/libero/libero"
BDDL_DIR="$LIBERO_INTERNAL_ROOT/bddl_files/libero_para"
INIT_DIR="$LIBERO_INTERNAL_ROOT/init_files/libero_para"
GOAL_BDDL_DIR="$LIBERO_INTERNAL_ROOT/bddl_files/libero_goal"

if [[ ! -d "$LIBERO_PARA_ROOT" ]]; then
  echo "[error] Missing LIBERO-Para root: $LIBERO_PARA_ROOT" >&2
  exit 1
fi

if [[ -x "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x /root/miniconda3/envs/openpi/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/openpi/bin/python
elif [[ -x /root/envs/openpi_plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi_plus/bin/python
elif [[ -x /root/envs/openpi-plus/bin/python ]]; then
  PYTHON_BIN=/root/envs/openpi-plus/bin/python
else
  echo "[error] Cannot find openpi python. Set PYTHON=/path/to/python." >&2
  exit 1
fi

if [[ ! -f "$TORCH_CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "[error] Missing PyTorch checkpoint: $TORCH_CHECKPOINT_DIR/model.safetensors" >&2
  echo "[error] Convert the JAX checkpoint first, for example:" >&2
  echo "[error]   $PYTHON_BIN examples/convert_jax_model_to_pytorch.py --checkpoint-dir /path/to/pi05_libero_jax_checkpoint --config-name pi05_libero --output-path $TORCH_CHECKPOINT_DIR --precision bfloat16" >&2
  exit 1
fi

if [[ ! -f "$TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" ]]; then
  echo "[error] Missing LIBERO norm stats: $TORCH_CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" >&2
  exit 1
fi

for required_dir in "$BDDL_DIR" "$INIT_DIR" "$GOAL_BDDL_DIR"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[error] Missing LIBERO-Para directory: $required_dir" >&2
    exit 1
  fi
done

mkdir -p "$LOG_ROOT" "$SERVER_LOG_ROOT" "$RESULT_ROOT" "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_INTERNAL_ROOT
bddl_files: $LIBERO_INTERNAL_ROOT/bddl_files
init_states: $LIBERO_INTERNAL_ROOT/init_files
assets: $LIBERO_INTERNAL_ROOT/assets
datasets: $LIBERO_PARA_ROOT/libero/datasets
YAML

export PYTHONPATH="$LIBERO_PARA_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset DISPLAY
unset XAUTHORITY

cd "$OPENPI_ROOT"

SERVER_PIDS=()
EVAL_PIDS=()
cleanup_server() {
  if (( ${#EVAL_PIDS[@]} > 0 )); then
    kill "${EVAL_PIDS[@]}" >/dev/null 2>&1 || true
    wait "${EVAL_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  EVAL_PIDS=()

  if (( ${#SERVER_PIDS[@]} > 0 )); then
    kill "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
    wait "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
  fi
  SERVER_PIDS=()
}
trap cleanup_server EXIT

start_server() {
  local seed="$1"
  local rank="$2"
  local server_log="$3"
  local gpu="${GPU_ARRAY[$rank]}"
  local rank_seed=$((seed + rank))

  CUDA_VISIBLE_DEVICES="$gpu" \
  OPENPI_POLICY_SEED="$rank_seed" \
  RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
  "$PYTHON_BIN" -c '
import os
import random
import runpy
import sys

seed = int(os.environ["OPENPI_POLICY_SEED"])
random.seed(seed)
try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass
try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
except Exception:
    pass

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
' scripts/serve_policy.py \
    --env LIBERO \
    --host "$SERVER_HOST" \
    --server-rank "$rank" \
    --server-world-size "$WORLD_SIZE" \
    --port "$PORT" \
    --pytorch-device cuda:0 \
    --require-cuda \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$TORCH_CHECKPOINT_DIR" \
    > "$server_log" 2>&1 &

  SERVER_PIDS[$rank]="$!"
}

wait_for_server() {
  local rank="$1"
  local server_log="$2"
  local server_pid="${SERVER_PIDS[$rank]}"
  local client_port=$((PORT + rank))
  local deadline=$((SECONDS + SERVER_WAIT_TIMEOUT_S))

  while (( SECONDS < deadline )); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "[error] Policy server rank=$rank exited early. Log: $server_log" >&2
      tail -n 80 "$server_log" >&2 || true
      return 1
    fi

    if "$PYTHON_BIN" - "$HOST" "$client_port" >/dev/null 2>&1 <<'PY'
import sys
import websockets.sync.client

host = sys.argv[1]
port = int(sys.argv[2])
with websockets.sync.client.connect(
    f"ws://{host}:{port}",
    open_timeout=3.0,
    close_timeout=1.0,
    max_size=None,
    ping_interval=None,
    ping_timeout=None,
) as ws:
    ws.recv()
PY
    then
      return 0
    fi

    sleep "$SERVER_WAIT_POLL_S"
  done

  echo "[error] Timed out waiting for policy server rank=$rank at $HOST:$client_port. Log: $server_log" >&2
  tail -n 80 "$server_log" >&2 || true
  return 1
}

merge_seed_results() {
  local seed_result_root="$1"
  local seed="$2"

  "$PYTHON_BIN" - "$seed_result_root" "$MODEL_NAME" "$seed" "$WORLD_SIZE" <<'PY'
import json
import pathlib
import re
import sys

seed_root = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
seed = int(sys.argv[3])
world_size = int(sys.argv[4])

rank_dirs = sorted(
    [path for path in seed_root.glob("rank_*") if path.is_dir()],
    key=lambda path: int(path.name.split("_", 1)[1]),
)
if not rank_dirs:
    raise SystemExit(f"No rank_* result directories found under {seed_root}")

eval_records = {}
rank_metas = []
for rank_dir in rank_dirs:
    meta_path = rank_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            rank_metas.append(json.load(f))

    for eval_path in sorted(rank_dir.glob("eval*.json")):
        if eval_path.name == "eval_unknown.json":
            continue
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        match = re.match(r"eval(\d+)\.json$", eval_path.name)
        eval_id = int(data.get("eval_id", match.group(1) if match else -1))
        merged = eval_records.setdefault(
            eval_id,
            {
                "eval_id": eval_id,
                "original_instruction": data.get("original_instruction"),
                "episodes": [],
            },
        )
        if merged["original_instruction"] is None:
            merged["original_instruction"] = data.get("original_instruction")
        merged["episodes"].extend(data.get("episodes", []))

if not eval_records:
    raise SystemExit(f"No eval*.json files found in rank outputs under {seed_root}")

total_episodes = 0
total_successes = 0
per_eval = {}
per_category = {}

for eval_id, data in sorted(eval_records.items()):
    data["episodes"].sort(key=lambda ep: (ep.get("task_id", -1), ep.get("variant_id", -1), ep.get("bddl_file", "")))
    out_path = seed_root / f"eval{eval_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    episodes = data["episodes"]
    successes = sum(1 for ep in episodes if ep.get("success", False))
    total_episodes += len(episodes)
    total_successes += successes
    per_eval[f"eval{eval_id}"] = {
        "total": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
    }

    for ep in episodes:
        if ep.get("paraphrase_type") == "comp":
            cat_key = f"comp_{'+'.join(ep.get('categories', []))}_{'+'.join(ep.get('subcategories', []))}"
        else:
            categories = ep.get("categories", [])
            subcategories = ep.get("subcategories", [])
            cat_key = (
                f"{ep.get('paraphrase_type')}_{categories[0]}_{subcategories[0]}"
                if categories and subcategories
                else "unknown"
            )
        stats = per_category.setdefault(cat_key, {"total": 0, "successes": 0})
        stats["total"] += 1
        stats["successes"] += int(bool(ep.get("success", False)))

for stats in per_category.values():
    stats["success_rate"] = stats["successes"] / stats["total"] if stats["total"] else 0.0

summary = {
    "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "per_eval": per_eval,
    "per_category": per_category,
}
with (seed_root / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

meta = {
    "model_name": model_name,
    "seed": seed,
    "eval_world_size": world_size,
    "rank_dirs": [path.name for path in rank_dirs],
    "rank_metas": rank_metas,
    "total_tasks": total_episodes,
}
with (seed_root / "meta.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"[info] Merged {total_episodes} episodes for seed={seed}: {total_successes}/{total_episodes}")
PY
}

write_summary() {
  "$PYTHON_BIN" - "$RESULT_ROOT" "$MODEL_NAME" <<'PY'
import json
import pathlib
import sys

result_root = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
rows = []

for summary_path in sorted(result_root.glob("seed*/summary.json")):
    seed_name = summary_path.parent.name
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    total = int(summary.get("total_episodes", 0))
    successes = int(summary.get("total_successes", 0))
    rate = 100.0 * successes / total if total else 0.0
    rows.append((seed_name, successes, total, rate))

print(f"# LIBERO-Para Summary: {model_name}")
print()
print("| Seed | Successes | Total | Success Rate |")
print("|------|-----------|-------|--------------|")
for seed_name, successes, total, rate in rows:
    print(f"| {seed_name} | {successes} | {total} | {rate:.1f}% |")

if rows:
    total_successes = sum(row[1] for row in rows)
    total_episodes = sum(row[2] for row in rows)
    total_rate = 100.0 * total_successes / total_episodes if total_episodes else 0.0
    mean_rate = sum(row[3] for row in rows) / len(rows)
    print(f"| all | {total_successes} | {total_episodes} | {total_rate:.1f}% |")
    print()
    print(f"Mean seed success rate: {mean_rate:.1f}%")
PY
}

echo "[info] OpenPI root: $OPENPI_ROOT"
echo "[info] LIBERO-Para root: $LIBERO_PARA_ROOT"
echo "[info] Python: $PYTHON_BIN"
echo "[info] Torch checkpoint: $TORCH_CHECKPOINT_DIR"
echo "[info] Seeds: $SEEDS"
echo "[info] GPUs: $GPUS, world size: $WORLD_SIZE, base server: $HOST:$PORT"
echo "[info] Logs: $LOG_ROOT"
echo "[info] Results: $RESULT_ROOT"

for seed in $SEEDS; do
  seed_log_root="$LOG_ROOT/seed${seed}"
  seed_server_log_root="$SERVER_LOG_ROOT/seed${seed}"
  seed_result_root="$RESULT_ROOT/seed${seed}"
  mkdir -p "$seed_log_root" "$seed_server_log_root" "$seed_result_root"

  SERVER_PIDS=()
  EVAL_PIDS=()
  echo "[info] Starting $WORLD_SIZE torch policy servers for seed=$seed"
  for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    server_log="$seed_server_log_root/server_rank_${rank}.log"
    start_server "$seed" "$rank" "$server_log"
  done

  for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    server_log="$seed_server_log_root/server_rank_${rank}.log"
    wait_for_server "$rank" "$server_log"
  done

  max_tasks_args=()
  if (( MAX_TASKS > 0 )); then
    max_tasks_args=(--max_tasks "$MAX_TASKS")
  fi

  echo "[info] Evaluating LIBERO-Para seed=$seed on $WORLD_SIZE ranks"
  for (( rank = 0; rank < WORLD_SIZE; rank++ )); do
    gpu="${GPU_ARRAY[$rank]}"
    rank_result_root="$seed_result_root/rank_${rank}"
    mkdir -p "$rank_result_root"

    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" LOCAL_RANK="$rank" WORLD_SIZE="$WORLD_SIZE" \
    "$PYTHON_BIN" "$LIBERO_PARA_ROOT/eval_scripts/examples/eval_pi05.py" \
      --host "$HOST" \
      --port "$PORT" \
      --eval_rank "$rank" \
      --eval_world_size "$WORLD_SIZE" \
      --resize_size "$RESIZE_SIZE" \
      --replan_steps "$REPLAN_STEPS" \
      --bddl_dir "$BDDL_DIR" \
      --init_dir "$INIT_DIR" \
      --goal_bddl_dir "$GOAL_BDDL_DIR" \
      --mode para \
      --model_name "$MODEL_NAME" \
      --gpu "$gpu" \
      --seed "$seed" \
      --max_steps "$MAX_STEPS" \
      --num_steps_wait "$NUM_STEPS_WAIT" \
      --output_dir "$rank_result_root" \
      "${max_tasks_args[@]}" \
      > "$seed_log_root/eval_rank_${rank}.log" 2>&1 &
    EVAL_PIDS+=("$!")
  done

  eval_status=0
  for pid in "${EVAL_PIDS[@]}"; do
    wait "$pid" || eval_status="$?"
  done
  EVAL_PIDS=()
  if (( eval_status != 0 )); then
    echo "[error] Evaluation failed for seed=$seed. Logs: $seed_log_root/eval_rank_<rank>.log" >&2
    exit "$eval_status"
  fi

  echo "[info] Finished LIBERO-Para seed=$seed"
  cleanup_server
  merge_seed_results "$seed_result_root" "$seed"
done

write_summary | tee "$LOG_ROOT/summary.md"

echo "[info] Results: $RESULT_ROOT"
echo "[info] Summary: $LOG_ROOT/summary.md"
echo "[info] Optional metrics:"
echo "[info]   $PYTHON_BIN $LIBERO_PARA_ROOT/metrics/analyze_results.py --model_path $RESULT_ROOT --model_name '$MODEL_NAME' --output_dir $LOG_ROOT/metrics"
