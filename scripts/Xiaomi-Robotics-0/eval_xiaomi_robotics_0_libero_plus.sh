#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Xiaomi-Robotics-0"
export XIAOMI_ROOT="$MODEL_ROOT"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/LIBERO-plus}"
export LOG_BASE="${LOG_BASE:-$ROOT/results/Xiaomi-Robotics-0/libero_plus}"
exec bash "$MODEL_ROOT/eval_libero_plus.sh" "$@"

