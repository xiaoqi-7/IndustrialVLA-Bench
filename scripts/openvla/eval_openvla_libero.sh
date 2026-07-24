#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/openvla"
export ROOT_DIR="$MODEL_ROOT"
export LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$ROOT/results/openvla_raw/$(date +%Y%m%d_%H%M%S)}"
exec bash "$MODEL_ROOT/eval.sh" "$@"

