#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MODEL_ROOT="$ROOT/models/Xiaomi-Robotics-0"
cd "$MODEL_ROOT"
exec bash scripts/launch_libero.sh "$@"

