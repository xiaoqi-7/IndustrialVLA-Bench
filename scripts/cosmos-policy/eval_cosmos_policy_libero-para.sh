#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/cosmos-policy"
export PARA_ROOT="${PARA_ROOT:-$ROOT/LIBERO-para}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/cosmos-policy/libero_para/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/run_libero_para.sh" "$@"

