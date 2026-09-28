#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/unifolm-vla"
export PROJECT_ROOT="$MODEL_ROOT"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$ROOT/LIBERO-para}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/unifolm-vla/libero_para/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/scripts/eval_scripts/eval_libero_para_multi_gpu.sh" "$@"

