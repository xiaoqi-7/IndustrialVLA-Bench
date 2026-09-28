#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Isaac-GR00T"
export REPO_DIR="$MODEL_ROOT"
export BENCHMARK_ROOT="$ROOT"
export LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$ROOT/LIBERO-para}"
export LOG_ROOT="${LOG_ROOT:-$ROOT/results/Isaac-GR00T/libero_para}"
exec bash "$MODEL_ROOT/scripts/eval_libero_para.sh" "$@"

