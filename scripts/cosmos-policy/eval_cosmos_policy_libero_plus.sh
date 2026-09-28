#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/cosmos-policy"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$ROOT/LIBERO-plus}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/cosmos-policy/libero_plus/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/run_libero_plus.sh" "$@"

