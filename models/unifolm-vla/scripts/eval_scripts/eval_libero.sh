#!/usr/bin/env bash
set -euo pipefail
export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export PATH=$MACA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$MACA_HOME/lib:$MACA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
# =========================
# Project paths
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export LIBERO_HOME="${LIBERO_HOME:-$(cd "${PROJECT_ROOT}/../../LIBERO" && pwd -P)}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHON="${PYTHON:-/root/envs/unifolm_libero/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

# =========================
# GPU settings
# =========================
# 你之前测试健康的卡是物理 GPU 2 / 3 / 6 / 7。
# 单卡跑时，建议先用物理 GPU 2。
# 注意：CUDA_VISIBLE_DEVICES=2 后，程序内部看到的是 cuda:0。
export CUDA_VISIBLE_DEVICES=6

# =========================
# HuggingFace / TensorFlow settings
# =========================
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=4

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# =========================
# MuJoCo / robosuite rendering
# =========================
# 沐曦环境不要走 EGL，会报 METAX_dri.so / EGL device display 错误。
# osmesa 在你当前节点上容易 segfault。
# 所以这里使用 Xvfb + glfw + llvmpipe 软件渲染。
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

unset MUJOCO_EGL_DEVICE_ID
unset EGL_DEVICE_ID
unset NVIDIA_VISIBLE_DEVICES
unset NVIDIA_DRIVER_CAPABILITIES

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

# =========================
# Model / eval settings
# =========================
your_ckpt="${your_ckpt:-}"                        # set to .../UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt
vlm_pretrained_path="${vlm_pretrained_path:-}"    # set to a local UnifoLM-VLM-Base download

task_suite_name=libero_object
num_trials_per_task=50
window_size=2
unnorm_key="libero_object_no_noops"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $5}')
step_name=$(echo "$your_ckpt" | awk -F'/' '{print $6}')
video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

mkdir -p "${video_out_path}"

echo "[DEBUG] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[DEBUG] LIBERO_HOME=${LIBERO_HOME}"
echo "[DEBUG] PYTHON=${PYTHON}"
echo "[DEBUG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DEBUG] MUJOCO_GL=${MUJOCO_GL}"
echo "[DEBUG] PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-}"
echo "[DEBUG] DISPLAY=${DISPLAY:-}"
echo "[DEBUG] video_out_path=${video_out_path}"

# =========================
# Check xvfb-run
# =========================
if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[ERROR] xvfb-run not found. Install it first:"
    echo "apt-get update && apt-get install -y xvfb libgl1-mesa-glx libgl1-mesa-dri libglfw3 mesa-utils"
    exit 1
fi

# =========================
# Tiny CUDA test
# =========================
"${PYTHON}" - <<'PY'
import torch
print("[DEBUG] cuda available:", torch.cuda.is_available(), flush=True)
print("[DEBUG] cuda count:", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("[DEBUG] device 0:", torch.cuda.get_device_name(0), flush=True)
    x = torch.ones((1,), device="cuda:0")
    torch.cuda.synchronize()
    print("[DEBUG] tiny cuda tensor ok", flush=True)
PY

# =========================
# Run eval
# =========================
echo "[INFO] Running LIBERO eval..."
echo "./experiments/LIBERO/eval_libero.py --args.pretrained-path ${your_ckpt} --args.vlm-pretrained-path ${vlm_pretrained_path} --args.task-suite-name ${task_suite_name} --args.num-trials-per-task ${num_trials_per_task} --args.video-out-path ${video_out_path} --args.unnorm-key ${unnorm_key} --args.window-size ${window_size}"

xvfb-run -a -s "-screen 0 1024x768x24" \
"${PYTHON}" -X faulthandler ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.vlm-pretrained-path "${vlm_pretrained_path}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.window-size "${window_size}"
