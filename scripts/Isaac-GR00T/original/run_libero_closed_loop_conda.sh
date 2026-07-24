#!/usr/bin/env bash
set -euo pipefail

############################
# User config
############################

REPO_DIR="${REPO_DIR:-/mnt/afs/zhengmingkai/raozf/benchmark/Isaac-GR00T}"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_libero}"

# suite: libero_10 / goal / object / spatial
SUITE="${SUITE:-libero_10}"

# 官方 N1.7 LIBERO checkpoint 根目录。
# 默认期望：
#   $MODEL_ROOT/libero_10
#   $MODEL_ROOT/libero_goal
#   $MODEL_ROOT/libero_object
#   $MODEL_ROOT/libero_spatial
MODEL_ROOT="${MODEL_ROOT:-/mnt/afs/zhengmingkai/raozf/models/GR00T-N1.7-libero}"

# 如果你直接指定 MODEL_PATH，则优先使用 MODEL_PATH。
MODEL_PATH="${MODEL_PATH:-}"

# 如果你已经把 Cosmos-Reason2-2B 下载在本地，可填这个路径。
# 脚本会把 checkpoint 配置里的 nvidia/Cosmos-Reason2-2B 替换成本地路径。
COSMOS_PATH="${COSMOS_PATH:-/mnt/afs/zhengmingkai/raozf/models/Cosmos-Reason2-2B}"
PATCH_COSMOS_PATH="${PATCH_COSMOS_PATH:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"

N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"

# debug 时可以只跑第一个任务：
#   ONLY_FIRST_TASK=1 bash run_libero_closed_loop_conda.sh
ONLY_FIRST_TASK="${ONLY_FIRST_TASK:-0}"

OFFLINE="${OFFLINE:-1}"

RUN_NAME="${RUN_NAME:-${SUITE}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/libero_closed_loop_conda}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
CLIENT_LOG_DIR="${LOG_DIR}/client"
SERVER_LOG="${LOG_DIR}/server.log"
SUMMARY_FILE="${LOG_DIR}/summary.txt"

############################
# Resolve suite/model path
############################

case "${SUITE}" in
  libero_10|10|long)
    SUITE="libero_10"
    DEFAULT_MODEL_PATH="${MODEL_ROOT}/libero_10"
    ;;
  goal|libero_goal)
    SUITE="goal"
    DEFAULT_MODEL_PATH="${MODEL_ROOT}/libero_goal"
    ;;
  object|libero_object)
    SUITE="object"
    DEFAULT_MODEL_PATH="${MODEL_ROOT}/libero_object"
    ;;
  spatial|libero_spatial)
    SUITE="spatial"
    DEFAULT_MODEL_PATH="${MODEL_ROOT}/libero_spatial"
    ;;
  *)
    echo "[ERROR] Unknown SUITE=${SUITE}. Use: libero_10, goal, object, spatial"
    exit 1
    ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$DEFAULT_MODEL_PATH"
fi

############################
# Activate conda
############################

cd "$REPO_DIR"
mkdir -p "$CLIENT_LOG_DIR"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh."
    exit 1
fi

conda activate "$CONDA_ENV"

############################
# Environment
############################

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# NVIDIA headless rendering
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset __GLX_VENDOR_LIBRARY_NAME || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

if [[ "$OFFLINE" == "1" ]]; then
    unset HF_ENDPOINT || true
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

############################
# Print config
############################

echo "========== GR00T LIBERO closed-loop conda eval =========="
echo "REPO_DIR             = $REPO_DIR"
echo "CONDA_ENV            = $CONDA_ENV"
echo "python               = $(which python)"
echo "SUITE                = $SUITE"
echo "MODEL_ROOT           = $MODEL_ROOT"
echo "MODEL_PATH           = $MODEL_PATH"
echo "COSMOS_PATH          = $COSMOS_PATH"
echo "PATCH_COSMOS_PATH    = $PATCH_COSMOS_PATH"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "MUJOCO_GL            = $MUJOCO_GL"
echo "PYOPENGL_PLATFORM    = $PYOPENGL_PLATFORM"
echo "HOST                 = $HOST"
echo "PORT                 = $PORT"
echo "N_EPISODES           = $N_EPISODES"
echo "N_ENVS               = $N_ENVS"
echo "N_ACTION_STEPS       = $N_ACTION_STEPS"
echo "MAX_EPISODE_STEPS    = $MAX_EPISODE_STEPS"
echo "ONLY_FIRST_TASK      = $ONLY_FIRST_TASK"
echo "OFFLINE              = $OFFLINE"
echo "LOG_DIR              = $LOG_DIR"
echo "=========================================================="

############################
# Sanity checks
############################

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Missing model config.json: $MODEL_PATH/config.json"
    exit 1
fi

if [[ ! -f "$REPO_DIR/gr00t/eval/run_gr00t_server.py" ]]; then
    echo "[ERROR] Missing server script: gr00t/eval/run_gr00t_server.py"
    exit 1
fi

if [[ ! -f "$REPO_DIR/gr00t/eval/rollout_policy.py" ]]; then
    echo "[ERROR] Missing rollout script: gr00t/eval/rollout_policy.py"
    exit 1
fi

python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
PY

############################
# Optional: patch Cosmos path
############################

if [[ "$PATCH_COSMOS_PATH" == "1" ]]; then
    if [[ -d "$COSMOS_PATH" ]]; then
        echo "[INFO] Patching nvidia/Cosmos-Reason2-2B to local COSMOS_PATH if found..."
        python - <<PY
from pathlib import Path

model_dir = Path("${MODEL_PATH}")
old = "nvidia/Cosmos-Reason2-2B"
new = "${COSMOS_PATH}"

for p in model_dir.rglob("*"):
    if p.is_file() and p.suffix in [".json", ".yaml", ".yml", ".txt"]:
        txt = p.read_text(errors="ignore")
        if old in txt:
            print("patch:", p)
            p.write_text(txt.replace(old, new))
PY
    else
        echo "[WARN] COSMOS_PATH does not exist: $COSMOS_PATH"
        echo "[WARN] Skip Cosmos path patch."
    fi
fi

if grep -R "nvidia/Cosmos-Reason2-2B" -n "$MODEL_PATH" >/dev/null 2>&1; then
    echo "[WARN] MODEL_PATH still contains remote Cosmos path:"
    grep -R "nvidia/Cosmos-Reason2-2B" -n "$MODEL_PATH" || true
fi

############################
# Task lists from official GR00T LIBERO README
############################

TASKS_LIBERO_10=(
"libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
"libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
"libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
"libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
"libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
"libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
"libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
"libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
"libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
"libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)

TASKS_GOAL=(
"libero_sim/open_the_middle_drawer_of_the_cabinet"
"libero_sim/put_the_bowl_on_the_stove"
"libero_sim/put_the_wine_bottle_on_top_of_the_cabinet"
"libero_sim/open_the_top_drawer_and_put_the_bowl_inside"
"libero_sim/put_the_bowl_on_top_of_the_cabinet"
"libero_sim/push_the_plate_to_the_front_of_the_stove"
"libero_sim/put_the_cream_cheese_in_the_bowl"
"libero_sim/turn_on_the_stove"
"libero_sim/put_the_bowl_on_the_plate"
"libero_sim/put_the_wine_bottle_on_the_rack"
)

TASKS_OBJECT=(
"libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
"libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket"
"libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket"
"libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
"libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket"
"libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
"libero_sim/pick_up_the_butter_and_place_it_in_the_basket"
"libero_sim/pick_up_the_milk_and_place_it_in_the_basket"
"libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
"libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket"
)

TASKS_SPATIAL=(
"libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
"libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)

case "$SUITE" in
  libero_10)
    TASKS=("${TASKS_LIBERO_10[@]}")
    ;;
  goal)
    TASKS=("${TASKS_GOAL[@]}")
    ;;
  object)
    TASKS=("${TASKS_OBJECT[@]}")
    ;;
  spatial)
    TASKS=("${TASKS_SPATIAL[@]}")
    ;;
esac

if [[ "$ONLY_FIRST_TASK" == "1" ]]; then
    TASKS=("${TASKS[0]}")
fi

############################
# Helpers
############################

wait_for_port() {
    local host="$1"
    local port="$2"
    local max_tries="${3:-240}"

    for i in $(seq 1 "$max_tries"); do
        if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
        then
            echo "[INFO] Policy server is ready at $host:$port"
            return 0
        fi

        if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
            echo "[ERROR] Policy server exited before ready."
            echo "Last 120 lines of server log:"
            tail -120 "$SERVER_LOG" || true
            return 1
        fi

        sleep 1
    done

    echo "[ERROR] Policy server not ready after ${max_tries}s."
    echo "Last 120 lines of server log:"
    tail -120 "$SERVER_LOG" || true
    return 1
}

cleanup() {
    local code=$?
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "[INFO] Stopping policy server PID=$SERVER_PID"
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        sleep 2
        kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    exit "$code"
}
trap cleanup EXIT

safe_name() {
    echo "$1" | sed 's#^libero_sim/##' | tr '/' '_' | tr ' ' '_' | tr -cd 'A-Za-z0-9_.-'
}

############################
# Start policy server
############################

echo "[INFO] Starting GR00T policy server..."

python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --port "$PORT" \
    > >(tee -a "$SERVER_LOG") 2>&1 &

SERVER_PID=$!
echo "[INFO] Server PID=$SERVER_PID"

wait_for_port "$HOST" "$PORT" 240

############################
# Run closed-loop rollout
############################

: > "$SUMMARY_FILE"

echo "[INFO] Running ${#TASKS[@]} task(s)."

for TASK in "${TASKS[@]}"; do
    NAME="$(safe_name "$TASK")"
    LOG_FILE="${CLIENT_LOG_DIR}/${NAME}.log"

    echo
    echo "============================================================"
    echo "[INFO] Running task: $TASK"
    echo "[INFO] Log file: $LOG_FILE"
    echo "============================================================"

    set +e
    python gr00t/eval/rollout_policy.py \
        --n-episodes "$N_EPISODES" \
        --policy-client-host "$HOST" \
        --policy-client-port "$PORT" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --env-name "$TASK" \
        --n-action-steps "$N_ACTION_STEPS" \
        --n-envs "$N_ENVS" \
        2>&1 | tee "$LOG_FILE"
    RET=${PIPESTATUS[0]}
    set -e

    if [[ "$RET" -ne 0 ]]; then
        echo "[ERROR] $TASK failed with code $RET" | tee -a "$SUMMARY_FILE"
    else
        echo "[OK] $TASK finished" | tee -a "$SUMMARY_FILE"
    fi

    {
        echo
        echo "----- $TASK -----"
        grep -Ei "success|succeed|rate|episode|result|score|completed|rollout" "$LOG_FILE" | tail -120 || true
    } >> "$SUMMARY_FILE"
done

############################
# Print summary
############################

echo
echo "===================== EVAL FINISHED ====================="
echo "Suite:   $SUITE"
echo "Model:   $MODEL_PATH"
echo "Logs:    $LOG_DIR"
echo "Summary: $SUMMARY_FILE"
echo "========================================================="
cat "$SUMMARY_FILE" || true
