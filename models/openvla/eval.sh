#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-GPU LIBERO evaluation for OpenVLA
#
# Example:
#   ./eval.sh \
#     --ckpt /mnt/afs/zhengmingkai/raozf/models/openvla/openvla-7b-finetuned-libero-10 \
#     --task_suite libero_10 \
#     --gpus "5 6 7" \
#     --num_trials_per_task 50 \
#     --center_crop True
# ============================================================
export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export PATH=$MACA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$MACA_HOME/lib:$MACA_HOME/lib64:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_DISABLED=true
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"

CKPT="openvla/openvla-7b-finetuned-libero-10"
TASK_SUITE="libero_10"
GPUS="5 6 7"
CENTER_CROP="True"
NUM_TRIALS_PER_TASK=50
NUM_STEPS_WAIT=10
MODEL_FAMILY="openvla"
USE_WANDB="False"
RUN_ID_NOTE=""
LOCAL_LOG_DIR=""

print_help() {
    cat <<'USAGE'
Usage:
  ./eval.sh [options]

Options:
  --ckpt PATH_OR_HF_ID              Model checkpoint path or HF repo id
  --task_suite NAME                 libero_spatial | libero_object | libero_goal | libero_10 | libero_90
  --gpus "0 1 2 3"                  GPU ids to use
  --center_crop True|False          Whether to use center crop
  --num_trials_per_task N           Number of trials per task
  --num_steps_wait N                Wait steps before policy actions
  --model_family NAME               Default: openvla
  --use_wandb True|False            Default: False
  --run_id_note NOTE                Extra note in log filename
  --local_log_dir DIR               Log directory
  -h, --help                        Show this help

Examples:
  ./eval.sh --ckpt /path/to/openvla-7b-finetuned-libero-10 --task_suite libero_10 --gpus "0 1 2 3"

  ./eval.sh --ckpt openvla/openvla-7b-finetuned-libero-10 --task_suite libero_spatial --gpus "0 1"

  ./eval.sh --ckpt /mnt/afs/zhengmingkai/raozf/models/openvla/openvla-7b-finetuned-libero-10 \
            --task_suite libero_10 \
            --gpus "0 1 2 3 4 5 6 7" \
            --num_trials_per_task 50 \
            --center_crop True
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --task_suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --center_crop)
            CENTER_CROP="$2"
            shift 2
            ;;
        --num_trials_per_task)
            NUM_TRIALS_PER_TASK="$2"
            shift 2
            ;;
        --num_steps_wait)
            NUM_STEPS_WAIT="$2"
            shift 2
            ;;
        --model_family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB="$2"
            shift 2
            ;;
        --run_id_note)
            RUN_ID_NOTE="$2"
            shift 2
            ;;
        --local_log_dir)
            LOCAL_LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            print_help
            exit 1
            ;;
    esac
done

cd "${ROOT_DIR}"

GPU_ARRAY=(${GPUS})
NUM_GPUS=${#GPU_ARRAY[@]}

if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "[ERROR] No GPU specified. Use --gpus \"0 1 2 3\""
    exit 1
fi

if [[ -z "${LOCAL_LOG_DIR}" ]]; then
    LOCAL_LOG_DIR="./experiments/logs/multigpu_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${LOCAL_LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/LIBERO:${ROOT_DIR}:${PYTHONPATH:-}"
# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa

# Disable Flax backend probing.
export TRANSFORMERS_NO_FLAX=1
export USE_FLAX=0

echo "============================================================"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MODEL_FAMILY: ${MODEL_FAMILY}"
echo "CKPT: ${CKPT}"
echo "TASK_SUITE: ${TASK_SUITE}"
echo "CENTER_CROP: ${CENTER_CROP}"
echo "NUM_TRIALS_PER_TASK: ${NUM_TRIALS_PER_TASK}"
echo "NUM_STEPS_WAIT: ${NUM_STEPS_WAIT}"
echo "GPUS: ${GPUS}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "LOCAL_LOG_DIR: ${LOCAL_LOG_DIR}"
echo "USE_WANDB: ${USE_WANDB}"
echo "RUN_ID_NOTE: ${RUN_ID_NOTE}"
echo "============================================================"

pids=()

for RANK in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$RANK]}"
    WORKER_LOG="${LOCAL_LOG_DIR}/worker_rank${RANK}_gpu${GPU_ID}.log"

    echo "[launch] rank=${RANK}/${NUM_GPUS}, gpu=${GPU_ID}, log=${WORKER_LOG}"

    EXTRA_NOTE="gpu${GPU_ID}"
    if [[ -n "${RUN_ID_NOTE}" ]]; then
        EXTRA_NOTE="${RUN_ID_NOTE}--gpu${GPU_ID}"
    fi

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python experiments/robot/libero/run_libero_eval.py \
        --model_family "${MODEL_FAMILY}" \
        --pretrained_checkpoint "${CKPT}" \
        --task_suite_name "${TASK_SUITE}" \
        --center_crop "${CENTER_CROP}" \
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
        --num_steps_wait "${NUM_STEPS_WAIT}" \
        --eval_rank "${RANK}" \
        --eval_world_size "${NUM_GPUS}" \
        --run_id_note "${EXTRA_NOTE}" \
        --local_log_dir "${LOCAL_LOG_DIR}" \
        --use_wandb "${USE_WANDB}" \
        > "${WORKER_LOG}" 2>&1 &

    pids+=($!)
done

echo "Launched ${#pids[@]} workers."

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "[ERROR] worker pid=${pid} failed"
        failed=1
    fi
done

echo "============================================================"
if [[ "${failed}" -eq 0 ]]; then
    echo "All workers finished successfully."
else
    echo "Some workers failed. Check logs in: ${LOCAL_LOG_DIR}"
fi
echo "Logs: ${LOCAL_LOG_DIR}"
echo "============================================================"

exit "${failed}"
