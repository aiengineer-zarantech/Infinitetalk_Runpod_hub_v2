#!/usr/bin/env bash
# Verify the 8 required models (names + exact byte sizes from models.list) under the pod models folder.
# Usage: check_models.sh [models_dir]      (default: $POD_MODELS_DIR or /workspace/models)
# Exit 0 = all 8 present with the right size. The optional Multi model is reported, never required.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${1:-${POD_MODELS_DIR:-/workspace/models}}"
bad=0
while IFS='|' read -r a b c d e; do
  case "$a" in ''|'#'*) continue ;; esac
  optional=0
  if [ "$a" = "optional" ]; then optional=1; folder="$b"; name="$c"; want="$d"; else folder="$a"; name="$b"; want="$c"; fi
  f="$MODELS/$folder/$name"
  if [ ! -f "$f" ]; then
    if [ "$optional" = 1 ]; then echo "  optional $folder/$name not present (multi-person jobs need it)"; continue; fi
    echo "  MISSING  $folder/$name"; bad=1; continue
  fi
  have=$(stat -c %s "$f")
  if [ "$have" != "$want" ]; then
    echo "  SIZE     $folder/$name: $have bytes, expected $want"; [ "$optional" = 1 ] || bad=1
  else
    echo "  OK       $folder/$name ($(( have / 1048576 )) MB)"
  fi
done < "$HERE/models.list"
if [ "$bad" = 0 ]; then echo "[MODELS] all 8 required models OK in $MODELS"; else
  echo "[MODELS] problem in $MODELS: run download_models.sh"; fi
exit "$bad"
