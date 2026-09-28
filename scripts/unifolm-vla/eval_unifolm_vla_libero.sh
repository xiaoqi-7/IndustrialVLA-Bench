#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/unifolm-vla"
export PROJECT_ROOT="$MODEL_ROOT"
export LIBERO_HOME="${LIBERO_HOME:-$ROOT/LIBERO}"
cd "$MODEL_ROOT"
exec bash scripts/eval_scripts/eval_libero.sh "$@"

