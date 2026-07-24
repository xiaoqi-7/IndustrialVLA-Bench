#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/unifolm-vla"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/LIBERO-plus}"
export LOG_BASE="${LOG_BASE:-$ROOT/results/unifolm-vla/libero_plus}"
exec bash "$MODEL_ROOT/scripts/eval_scripts/eval_libero_plus.sh" "$@"

