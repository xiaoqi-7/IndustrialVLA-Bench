#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Isaac-GR00T"
export REPO_DIR="$MODEL_ROOT"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/LIBERO-plus}"
export LOG_ROOT="${LOG_ROOT:-$ROOT/results/Isaac-GR00T/libero_plus}"
exec bash "$MODEL_ROOT/eval_libero_plus_suite.sh" "$@"

