#!/usr/bin/env bash
# First-time download of the 8 required models from their public huggingface.co sources into
# <models_dir>/<folder>/<exact file name>. Safe to re-run: files with the right size are skipped, a partial
# download resumes (curl -C -). Each file is checked against the byte size in models.list.
# Usage: download_models.sh [models_dir]   env: INCLUDE_MULTI=1 also gets the multi-person model,
#        HF_TOKEN=<read token> only if a source ever needs auth (all are public today).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${1:-${POD_MODELS_DIR:-/workspace/models}}"
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
need=0
while IFS='|' read -r a b c d e; do
  case "$a" in ''|'#'*) continue ;; esac
  if [ "$a" = "optional" ]; then
    [ "${INCLUDE_MULTI:-0}" = 1 ] || continue
    folder="$b"; name="$c"; want="$d"; url="$e"
  else folder="$a"; name="$b"; want="$c"; url="$d"; fi
  need=$(( need + want ))
done < "$HERE/models.list"
free=$(df -B1 --output=avail "$(dirname "$MODELS")" 2>/dev/null | tail -1 || echo 0)
echo "[DOWNLOAD] target $MODELS, up to $(( need / 1073741824 )) GB needed, $(( free / 1073741824 )) GB free"

while IFS='|' read -r a b c d e; do
  case "$a" in ''|'#'*) continue ;; esac
  if [ "$a" = "optional" ]; then
    [ "${INCLUDE_MULTI:-0}" = 1 ] || continue
    folder="$b"; name="$c"; want="$d"; url="$e"
  else folder="$a"; name="$b"; want="$c"; url="$d"; fi
  dir="$MODELS/$folder"; f="$dir/$name"; part="$f.part"
  mkdir -p "$dir"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$want" ]; then echo "  skip     $folder/$name (already OK)"; continue; fi
  echo "  download $folder/$name ($(( want / 1048576 )) MB)"
  curl -fL --retry 8 --retry-all-errors --retry-delay 5 -C - "${AUTH[@]}" -o "$part" "$url"
  have=$(stat -c %s "$part")
  [ "$have" = "$want" ] || { echo "  SIZE MISMATCH $folder/$name: $have != $want (kept $part, re-run to resume)"; exit 1; }
  mv -f "$part" "$f"
done < "$HERE/models.list"
exec bash "$HERE/check_models.sh" "$MODELS"
