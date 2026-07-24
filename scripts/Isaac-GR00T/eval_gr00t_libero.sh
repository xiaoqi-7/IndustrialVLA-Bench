#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Isaac-GR00T"
export REPO_DIR="$MODEL_ROOT"
export LOG_ROOT="${LOG_ROOT:-$ROOT/results/Isaac-GR00T/libero}"
exec bash "$MODEL_ROOT/run_libero_closed_loop_conda.sh" "$@"

