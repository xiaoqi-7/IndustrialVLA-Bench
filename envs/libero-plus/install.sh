#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install -r "$HERE/requirements.txt"
if [[ -s "$HERE/extra_requirements.txt" ]]; then
  "$PYTHON_BIN" -m pip install -r "$HERE/extra_requirements.txt"
fi
"$PYTHON_BIN" -m pip install -e "$ROOT/LIBERO-plus"

