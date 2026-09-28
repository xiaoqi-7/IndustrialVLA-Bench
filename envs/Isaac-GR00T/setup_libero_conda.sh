#!/usr/bin/env bash
set -euo pipefail
set -x

############################
# User config
############################

REPO_DIR="${REPO_DIR:-/mnt/afs/zhengmingkai/raozf/benchmark/Isaac-GR00T}"
CONDA_ENV="${CONDA_ENV:-/root/envs/gr00t_libero}"

# 默认不重装 torch，避免破坏当前 CUDA / GR00T 环境。
# 如果你的环境里根本没有 torch，可以运行：
#   INSTALL_TORCH=1 bash setup_libero_conda.sh
INSTALL_TORCH="${INSTALL_TORCH:-0}"

# N1.7 / Qwen3-VL 建议 transformers 较新。
INSTALL_TRANSFORMERS="${INSTALL_TRANSFORMERS:-1}"

# 如果没有网络，且你已经手动放好了 external_dependencies/LIBERO，
# 可以设置 ALLOW_CLONE=0，避免脚本尝试 git clone。
ALLOW_CLONE="${ALLOW_CLONE:-1}"

# NVIDIA headless rendering
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
unset __GLX_VENDOR_LIBRARY_NAME || true

############################
# Activate conda
############################

cd "$REPO_DIR"

echo "[INFO] REPO_DIR=$REPO_DIR"
echo "[INFO] CONDA_ENV=$CONDA_ENV"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh."
    echo "请先手动执行："
    echo "  conda activate /root/envs/gr00t_libero"
    exit 1
fi

conda activate "$CONDA_ENV"

which python
python --version
which pip

############################
# Basic checks
############################

python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
PY

if python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    echo "[INFO] Existing torch:"
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
else
    echo "[WARN] torch is not installed or cannot be imported."
    if [[ "$INSTALL_TORCH" != "1" ]]; then
        echo "[ERROR] torch 不可用，但 INSTALL_TORCH=0。"
        echo "请用：INSTALL_TORCH=1 bash setup_libero_conda.sh"
        exit 1
    fi
fi

############################
# Prepare LIBERO dependency
############################

LIBERO_REPO="$REPO_DIR/external_dependencies/LIBERO"

mkdir -p "$REPO_DIR/external_dependencies"

if [[ -d "$LIBERO_REPO" && -f "$LIBERO_REPO/requirements.txt" ]]; then
    echo "[INFO] LIBERO already exists: $LIBERO_REPO"
else
    echo "[INFO] Preparing LIBERO repo at $LIBERO_REPO"

    if [[ -f ".gitmodules" ]] && grep -q "external_dependencies/LIBERO" .gitmodules; then
        echo "[INFO] Found LIBERO submodule entry. Updating by relative path."
        git submodule sync --recursive || true
        git submodule update --init --recursive external_dependencies/LIBERO || {
            echo "[WARN] git submodule update failed."
            if [[ "$ALLOW_CLONE" == "1" ]]; then
                echo "[INFO] Fallback to direct clone."
                rm -rf "$LIBERO_REPO"
                git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_REPO"
            else
                echo "[ERROR] ALLOW_CLONE=0 and submodule update failed."
                exit 1
            fi
        }
    else
        echo "[WARN] .gitmodules missing LIBERO entry."
        if [[ "$ALLOW_CLONE" == "1" ]]; then
            echo "[INFO] Direct clone LIBERO."
            rm -rf "$LIBERO_REPO"
            git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_REPO"
        else
            echo "[ERROR] ALLOW_CLONE=0 and LIBERO is missing."
            exit 1
        fi
    fi
fi

test -d "$LIBERO_REPO"
test -f "$LIBERO_REPO/requirements.txt"

echo "[INFO] LIBERO_REPO=$LIBERO_REPO"

############################
# Install Python dependencies
############################

python -m pip install -U pip setuptools wheel

# 避免 LIBERO requirements 里可能覆盖 torch / torchvision / numpy。
REQ_NO_TORCH="/tmp/libero_requirements_no_torch_$$.txt"
grep -v -E '^[[:space:]]*(torch|torchvision|torchaudio|numpy)([<=>[:space:]]|$)' "$LIBERO_REPO/requirements.txt" > "$REQ_NO_TORCH" || true

echo "[INFO] Installing LIBERO requirements without torch/torchvision/torchaudio/numpy..."
python -m pip install -r "$REQ_NO_TORCH"
rm -f "$REQ_NO_TORCH"

echo "[INFO] Installing LIBERO editable..."
python -m pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat

echo "[INFO] Installing Isaac-GR00T editable without deps..."
python -m pip install -e "$REPO_DIR" --no-deps


if [[ "$INSTALL_TRANSFORMERS" == "1" ]]; then
    echo "[INFO] Installing transformers for GR00T N1.7 / Qwen3-VL..."
    python -m pip install transformers==4.57.3 accelerate safetensors
fi

if [[ "$INSTALL_TORCH" == "1" ]]; then
    echo "[INFO] Installing torch/torchvision. This may change your CUDA torch stack."
    python -m pip install torch==2.5.1 torchvision==0.20.1
fi

# 再安装一次 GR00T editable，确保当前 repo 优先。
python -m pip install -e "$REPO_DIR" --no-deps

############################
# Verify imports
############################

python - <<'PY'
import sys
print("Python:", sys.executable)

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import transformers
print("transformers:", transformers.__version__)

import gymnasium
print("gymnasium:", gymnasium.__version__)

import libero
print("libero import OK")
PY

############################
# Reset LIBERO config and register envs
############################

rm -rf "$HOME/.libero"

# 这里 printf 'n\n' 是为了自动回答 LIBERO 初始化时的交互问题。
printf 'n\n' | python - <<'PY'
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
print("register_libero_envs OK")
PY

############################
# Test one LIBERO env
############################

python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()

import gymnasium as gym

env_name = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
print("Testing env:", env_name)

env = gym.make(env_name)
obs, info = env.reset()
print("Env reset OK")
print("obs type:", type(obs))

try:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Env step OK")
except Exception as e:
    print("[WARN] Env step failed, but reset succeeded:", repr(e))

env.close()
print("[OK] LIBERO conda setup finished.")
PY
