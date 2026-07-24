#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/cosmos-policy"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PACKAGE_ROOT="${LIBERO_PACKAGE_ROOT:-$ROOT/LIBERO}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/cosmos-policy/libero/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/run_libero.sh" "$@"

