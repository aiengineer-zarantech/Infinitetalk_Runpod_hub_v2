#!/usr/bin/env bash
# Pod start (RunPod "Container Start Command", image v32 unchanged). Branch deploy/pod.
#   1. /runpod-volume -> /workspace (so every /runpod-volume path in the worker code points at the pod volume)
#   2. overlay this repo's worker code onto the v32 image (no rebuild)
#   3. verify the 8 models in /workspace/models (no copy: MODEL_SOURCE=baked links them in place)
#   4. compile cache in /workspace/cache/compile
#   5. /entrypoint.sh as on serverless (GPU guard, model links, ComfyUI), then the pod job API on :8000
set -euo pipefail
REPO="${POD_REPO:-/workspace/Infinitetalk_Runpod_hub_v2}"
MODELS="${POD_MODELS_DIR:-/workspace/models}"
V32_HANDLER_SHA=07b8f9bc4574aff6990ce7e1bbb50d314fd8eeda5d39fdfdd99e0deed28d56fe
echo "=== [POD] start_pod.sh ($(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo 'no git'), $(date -u +%FT%TZ))"

# 1. volume link
[ -d /workspace ] || { echo "❌ [POD] /workspace not mounted: attach the network volume to the pod"; exit 1; }
if [ -L /runpod-volume ]; then
  ln -sfn /workspace /runpod-volume
elif [ -d /runpod-volume ]; then
  if rmdir /runpod-volume 2>/dev/null; then ln -s /workspace /runpod-volume
  else echo "⚠️ [POD] /runpod-volume is a non-empty folder, left as is (worker paths may not see /workspace)"; fi
else
  ln -s /workspace /runpod-volume
fi
echo "[POD] /runpod-volume -> $(readlink -f /runpod-volume)"

# 2. code overlay (handler, setup_models, entrypoint, worker_guard, pod API)
have=$(sha256sum /handler.py | cut -c1-64)
mine=$(sha256sum "$REPO/handler.py" | cut -c1-64)
if [ "$have" != "$V32_HANDLER_SHA" ] && [ "$have" != "$mine" ]; then
  echo "⚠️ [POD] /handler.py is neither v32 nor this repo's: the image is not v32, overlay may not fit"
fi
cp -f "$REPO/handler.py" "$REPO/setup_models.py" "$REPO/entrypoint.sh" /
cp -f "$REPO"/worker_guard/*.py /worker_guard/
mkdir -p /pod && cp -f "$REPO/pod/pod_api.py" /pod/
# MODEL_LOCK: locked config + the zz_worker_mem node (clear non-model memory without unloading the model)
cp -f "$REPO/locked_model_config.json" /
mkdir -p /ComfyUI/custom_nodes/zz_worker_mem && cp -f "$REPO/comfy_mem/__init__.py" /ComfyUI/custom_nodes/zz_worker_mem/
sed -i 's/\r$//' /handler.py /setup_models.py /entrypoint.sh /worker_guard/*.py /pod/pod_api.py
chmod 755 /entrypoint.sh /setup_models.py
rm -rf /worker_guard/__pycache__ /__pycache__
echo "[POD] worker code from $REPO copied over the image"

# 3. models: verify, never copy
bash "$REPO/pod/check_models.sh" "$MODELS" || {
  echo "❌ [POD] models missing or wrong size. Run once: bash $REPO/pod/download_models.sh"
  [ "${POD_ALLOW_MISSING_MODELS:-0}" = "1" ] || { echo "[POD] waiting for setup (sleep infinity); fix and restart the pod"; exec sleep infinity; }
}
export MODEL_SOURCE=baked BAKED_MODELS_DIR="$MODELS" HF_CACHE_FALLBACK=0
export UPSCALE_MODELS_AT_STARTUP="${UPSCALE_MODELS_AT_STARTUP:-0}"

# 4. compile cache + volume check on /workspace
export COMPILE_CACHE_VOLUME="${COMPILE_CACHE_VOLUME:-/workspace}" VOLUME_ROOT_CHECK=/workspace
mkdir -p /workspace/cache/compile

# 5. pod job API instead of the serverless loop (entrypoint.sh reads POD_MODE)
export POD_MODE=1 POD_API="${POD_API:-pod}" POD_API_HOST="${POD_API_HOST:-0.0.0.0}" POD_API_PORT="${POD_API_PORT:-8000}"
TOKEN_NOW="${POD_API_TOKEN:-}"
if [ "$POD_API" = "pod" ] && [ "${#TOKEN_NOW}" -lt 16 ]; then
  echo "❌ [POD] POD_API_TOKEN (16+ chars) is not set in the pod env; the API refuses to start without it"
  exec sleep infinity
fi
exec /entrypoint.sh
