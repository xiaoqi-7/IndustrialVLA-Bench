#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Xiaomi-Robotics-0"
export XIAOMI_ROOT="$MODEL_ROOT"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$ROOT/LIBERO-para}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/Xiaomi-Robotics-0/libero_para/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/scripts/eval_libero_para.sh" "$@"

