#!/usr/bin/env bash
# Upload the 8 models of the I2V single workflow to ONE private Hugging Face repo, in the folder layout
# setup_models.py (MODEL_SOURCE=hf_cache) expects. RunPod allows one cached model (repo) per endpoint.
#
# Run it on a cheap RunPod CPU pod in the same datacenter as the network volume, with the volume attached, so the
# files are the exact ones the workers use today and nothing crosses your home connection:
#   pip install -U "huggingface_hub[hf_xet]"
#   hf auth login                       # a WRITE token; never paste it into this script
#   bash upload_hf_cache.sh <org>/infinitetalk-models [/runpod-volume]
#
# Checks every file size against the public source before uploading (FORCE=1 uploads a mismatch anyway).
# Re-running is safe: hf upload skips files the repo already has with the same content.
set -euo pipefail

REPO="${1:?usage: upload_hf_cache.sh <org>/<name> [search_root]}"
ROOT="${2:-/runpod-volume}"
command -v hf >/dev/null || { echo "hf CLI not found: pip install -U \"huggingface_hub[hf_xet]\""; exit 1; }

# path in repo | size in bytes (public source on huggingface.co, checked 2026-11-02)
FILES=(
  "diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors|16993877896"
  "diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors|2713548210"
  "diffusion_models/MelBandRoformer_fp16.safetensors|456479072"
  "diffusion_models/wav2vec2-chinese-base_fp16.safetensors|190115368"
  "text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors|6731333792"
  "clip_vision/clip_vision_h.safetensors|1264219396"
  "vae/Wan2_1_VAE_bf16.safetensors|253806278"
  "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors|738005744"
)

echo "== 1/3 finding the files under $ROOT"
declare -A SRC
missing=0
for entry in "${FILES[@]}"; do
  path="${entry%%|*}"; want="${entry##*|}"; name="$(basename "$path")"
  found="$(find "$ROOT" -type f -name "$name" -not -name '*.tmp' 2>/dev/null | head -1 || true)"
  if [ -z "$found" ]; then
    echo "  MISSING  $name"; missing=1; continue
  fi
  have="$(stat -c %s "$found")"
  if [ "$have" != "$want" ]; then
    echo "  SIZE     $name: $have bytes, public source has $want"
    [ "${FORCE:-0}" = "1" ] || missing=1
  else
    echo "  OK       $name ($have bytes) $found"
  fi
  SRC["$path"]="$found"
done
[ "$missing" = "0" ] || { echo "Stopping: fix the files above (FORCE=1 uploads size mismatches anyway)."; exit 1; }

echo "== 2/3 private repo $REPO"
hf repo create "$REPO" --private --exist-ok >/dev/null 2>&1 || hf repo create "$REPO" --private || true

echo "== 3/3 uploading (largest file is ~17 GB; hf_xet resumes an interrupted upload)"
for entry in "${FILES[@]}"; do
  path="${entry%%|*}"
  echo "  -> $path"
  hf upload "$REPO" "${SRC[$path]}" "$path" --repo-type model --commit-message "add $path"
done

echo "Done. RunPod endpoint: Model = $REPO (with a read token), env MODEL_SOURCE=hf_cache, HF_CACHE_REPO=$REPO"
