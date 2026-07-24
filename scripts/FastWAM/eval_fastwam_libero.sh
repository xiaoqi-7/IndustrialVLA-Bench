#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/FastWAM"
export BENCHMARK_ROOT="$ROOT"
export FASTWAM_SOURCE_ROOT="${FASTWAM_SOURCE_ROOT:-$MODEL_ROOT}"
export LIBERO_ROOT="${LIBERO_ROOT:-$ROOT/LIBERO}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/FastWAM/libero/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/run_fastwam_libero.sh" "$@"

