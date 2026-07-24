#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ARCHIVE="${ARCHIVE:-$ROOT/dataset/assets.zip}"
ASSETS_PARENT="$ROOT/libero/libero"
ASSETS_DIR="$ASSETS_PARENT/assets"
STAMP="$(date +%Y%m%d_%H%M%S)"
NEW_DIR="$ASSETS_PARENT/assets.extracting.$STAMP"
BACKUP_DIR="$ASSETS_PARENT/assets.standard.$STAMP"

[[ -f "$ARCHIVE" ]] || {
  echo "[error] Missing archive: $ARCHIVE" >&2
  echo "[info] Download assets.zip from:" >&2
  echo "       https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main" >&2
  echo "[info] Then run: ARCHIVE=/path/to/assets.zip bash prepare_assets.sh" >&2
  exit 1
}
command -v bsdtar >/dev/null 2>&1 || {
  echo "[error] bsdtar is required to extract the official archive." >&2
  exit 1
}
[[ ! -e "$NEW_DIR" ]] || { echo "[error] Path already exists: $NEW_DIR" >&2; exit 1; }

mkdir -p "$NEW_DIR"
echo "[info] Extracting $ARCHIVE"
echo "[info] Temporary destination: $NEW_DIR"
# Archive members start with
# inspire/.../dataset/LIBERO-plus-0/assets/; stripping 11 components places
# scenes/, new_objects/, textures/, etc. directly under NEW_DIR.
bsdtar -xf "$ARCHIVE" --strip-components 11 -C "$NEW_DIR"

[[ -d "$NEW_DIR/new_objects" ]] || {
  echo "[error] Extraction did not produce new_objects/. Kept: $NEW_DIR" >&2
  exit 1
}

if [[ -e "$ASSETS_DIR" ]]; then
  mv -- "$ASSETS_DIR" "$BACKUP_DIR"
fi
mv -- "$NEW_DIR" "$ASSETS_DIR"
echo "[ok] Full LIBERO-plus assets installed: $ASSETS_DIR"
[[ -e "$BACKUP_DIR" ]] && echo "[info] Previous assets kept at: $BACKUP_DIR"
