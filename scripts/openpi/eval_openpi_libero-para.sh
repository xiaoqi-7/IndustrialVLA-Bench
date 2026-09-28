#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/openpi"
export OPENPI_ROOT="$MODEL_ROOT"
export LIBERO_PARA_ROOT="${LIBERO_PARA_ROOT:-$ROOT/LIBERO-para}"
export RUN_NAME="${RUN_NAME:-torch_libero_para_$(date +%Y%m%d_%H%M%S)}"
export LOG_ROOT="${LOG_ROOT:-$ROOT/results/openpi/libero_para/$RUN_NAME}"
export RESULT_ROOT="${RESULT_ROOT:-$LOG_ROOT/results}"
exec bash "$MODEL_ROOT/eval_libero_para.sh" "$@"

