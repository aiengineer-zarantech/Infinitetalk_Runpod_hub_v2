#!/usr/bin/env python3
import os
import sys

COMFY_MODELS_DIR = "/ComfyUI/models"

MODEL_DESTINATIONS = {
    "Wan2_1_VAE_bf16.safetensors": "vae",
    "clip_vision_h.safetensors": "clip_vision",
    "umt5-xxl-enc-fp8_e4m3fn.safetensors": "text_encoders",
    "MelBandRoformer_fp16.safetensors": "diffusion_models",
    "wav2vec2-chinese-base_fp16.safetensors": "diffusion_models",
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors": "loras",
    "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors": "diffusion_models",
    "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors": "diffusion_models",
    "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors": "diffusion_models",
}

def scan_and_link():
    print("=" * 60)
    print("🔍 [SETUP_MODELS] Starting model discovery and linking...")
    print("=" * 60)

    import subprocess
    try:
        print("\n💾 System Mounts & Disk Space:")
        print(subprocess.getoutput("df -h"))
        print("\n🔍 Mount points:")
        print(subprocess.getoutput("grep -E 'volume|workspace|s3|fuse|nfs' /proc/mounts || head -n 10 /proc/mounts"))
    except Exception as me:
        print(f"Error checking mounts: {me}")

    # 1. Direct standard path checks and symlinking (bypasses GeeseFS directory readdir limitations)
    print("\n🎯 Verifying / Symlinking required models:")
    missing_models = []

    for model_name, subfolder in MODEL_DESTINATIONS.items():
        dest_dir = os.path.join(COMFY_MODELS_DIR, subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, model_name)

        # Candidate paths in volume/workspace
        candidate_paths = [
            os.path.join("/runpod-volume/models", subfolder, model_name),
            os.path.join("/runpod-volume", subfolder, model_name),
            os.path.join("/runpod-volume", model_name),
            os.path.join("/workspace/models", subfolder, model_name),
            os.path.join("/workspace", subfolder, model_name),
            os.path.join("/workspace", model_name),
        ]

        src_found = None
        for cand in candidate_paths:
            try:
                if os.path.exists(cand):
                    src_found = cand
                    break
            except Exception:
                pass

        if src_found:
            try:
                if os.path.islink(dest_path) or os.path.exists(dest_path):
                    os.remove(dest_path)
                os.symlink(src_found, dest_path)
                size_mb = os.path.getsize(dest_path) / (1024 * 1024)
                print(f"   ✅ [LINKED] {model_name} ({size_mb:.1f} MB) -> {dest_path}")
            except Exception as e:
                print(f"   ❌ Failed to symlink {model_name}: {e}")
        else:
            missing_models.append((model_name, subfolder, dest_path))

    # 2. Handle missing models (auto-download helper audio models if needed)
    if missing_models:
        print("\n⚠️ Handling missing models:")
        auto_download_urls = {
            "MelBandRoformer_fp16.safetensors": "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors",
            "wav2vec2-chinese-base_fp16.safetensors": "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
        }

        for model_name, subfolder, dest_path in missing_models:
            if model_name in auto_download_urls:
                url = auto_download_urls[model_name]
                print(f"   ⬇️ [AUTO-DOWNLOAD] Downloading {model_name}...")
                vol_target = os.path.join("/runpod-volume/models", subfolder, model_name)
                save_path = vol_target if os.path.exists("/runpod-volume/models") else dest_path
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    import urllib.request
                    urllib.request.urlretrieve(url, save_path)
                    print(f"   ✅ Downloaded {model_name} to {save_path}")
                    if save_path != dest_path:
                        if os.path.islink(dest_path) or os.path.exists(dest_path):
                            os.remove(dest_path)
                        os.symlink(save_path, dest_path)
                except Exception as de:
                    print(f"   ❌ Failed auto-download for {model_name}: {de}")
            else:
                print(f"   ❌ [CRITICAL MISSING] {model_name} (expected in {dest_path})")

    # 3. Summary of ComfyUI model directories
    print("\n📊 Final ComfyUI models directory status:")
    if os.path.exists(COMFY_MODELS_DIR):
        for sub in sorted(os.listdir(COMFY_MODELS_DIR)):
            sub_path = os.path.join(COMFY_MODELS_DIR, sub)
            if os.path.isdir(sub_path):
                models_in_sub = [f for f in os.listdir(sub_path) if not f.startswith("put_")]
                if models_in_sub:
                    print(f"   📁 {sub}: {models_in_sub}")
    print("=" * 60)

if __name__ == "__main__":
    scan_and_link()
