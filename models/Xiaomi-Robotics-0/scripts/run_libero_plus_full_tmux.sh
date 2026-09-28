#!/usr/bin/env bash
set -euo pipefail

XIAOMI_ROOT="/mnt/afs/zhengmingkai/raozf/benchmark/Xiaomi-Robotics-0"
MODEL_PATH="${MODEL_PATH:-/mnt/afs/zhengmingkai/raozf/models/Xiaomi-Robotics-0-LIBERO}"
LOG_BASE="${LOG_BASE:-$XIAOMI_ROOT/logs}"
NUM_PORTS="${NUM_PORTS:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORLD_SIZE="${WORLD_SIZE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
RUN_NAME="${RUN_NAME:-xiaomi_libero_plus_full_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-10086}"
SESSION_NAME="${SESSION_NAME:-$RUN_NAME}"

cd "$XIAOMI_ROOT"
mkdir -p "$LOG_BASE/libero_plus_eval"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux not found. Please install tmux or run scripts/eval_libero_plus.sh directly." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
  echo "[error] tmux session already exists: $SESSION_NAME" >&2
  echo "[hint] tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

python_bin="/root/envs/xiaomi_plus/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "[error] Missing Python: $python_bin" >&2
  exit 1
fi

"$python_bin" - "$BASE_PORT" "$WORLD_SIZE" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
world_size = int(sys.argv[2])
busy = []
for port in range(base_port, base_port + world_size):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()

if busy:
    for port, error in busy:
        print(f"[error] Port 127.0.0.1:{port} is busy: {error}", file=sys.stderr)
    sys.exit(1)
PY

top_log="$LOG_BASE/libero_plus_eval/${RUN_NAME}.out"
launcher="$LOG_BASE/libero_plus_eval/${RUN_NAME}.launcher.sh"
run_dir="$LOG_BASE/libero_plus_eval/$RUN_NAME"

cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$XIAOMI_ROOT"
export GPUS="$GPUS"
export WORLD_SIZE="$WORLD_SIZE"
export NUM_TRIALS_PER_TASK="$NUM_TRIALS_PER_TASK"
export TASK_SUITES="$TASK_SUITES"
export RUN_NAME="$RUN_NAME"
export BASE_PORT="$BASE_PORT"
bash scripts/eval_libero_plus.sh "$MODEL_PATH" "$NUM_PORTS" "$NUM_GPUS" "$LOG_BASE" 2>&1 | tee "$top_log"
EOF
chmod +x "$launcher"

tmux new-session -d -s "$SESSION_NAME" "bash '$launcher'"

echo "[info] Started Xiaomi LIBERO-plus full eval in tmux."
echo "[info] RUN_NAME: $RUN_NAME"
echo "[info] tmux: tmux attach -t $SESSION_NAME"
echo "[info] log: tail -f $top_log"
echo "[info] run dir: $run_dir"
echo "[info] summary: $run_dir/summary.md"
