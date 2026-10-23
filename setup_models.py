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

    # 1. Inspect search paths
    search_roots = ["/runpod-volume", "/workspace", "/mnt", "/root"]
    found_files = {}

    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            print(f"ℹ️ Search root does not exist: {root_dir}")
            continue

        print(f"\n📂 Inspecting {root_dir}:")
        try:
            entries = os.listdir(root_dir)
            print(f"   Direct contents ({len(entries)} items): {entries[:20]}")
        except Exception as e:
            print(f"   Could not list {root_dir}: {e}")

        # Recursive scan for safetensors/bin/pt
        for root, dirs, files in os.walk(root_dir, followlinks=True):
            for file in files:
                if file.endswith((".safetensors", ".bin", ".pt", ".pth")):
                    full_path = os.path.join(root, file)
                    found_files[file] = full_path
                    print(f"   Found model: {file} -> {full_path}")

    # 2. Check standard directory links
    volume_models_dir = "/runpod-volume/models"
    if os.path.exists(volume_models_dir):
        print(f"\n🔗 Linking from standard {volume_models_dir}:")
        for sub in os.listdir(volume_models_dir):
            src_sub = os.path.join(volume_models_dir, sub)
            if os.path.isdir(src_sub):
                dest_sub = os.path.join(COMFY_MODELS_DIR, sub)
                os.makedirs(dest_sub, exist_ok=True)
                for item in os.listdir(src_sub):
                    src_file = os.path.join(src_sub, item)
                    dest_file = os.path.join(dest_sub, item)
                    if not os.path.exists(dest_file):
                        try:
                            os.symlink(src_file, dest_file)
                            print(f"   Linked {item} to {dest_sub}")
                        except Exception as e:
                            print(f"   Failed to link {src_file}: {e}")

    # 3. Direct matching and symlinking for known models
    print("\n🎯 Verifying / Symlinking required models:")
    for model_name, subfolder in MODEL_DESTINATIONS.items():
        dest_dir = os.path.join(COMFY_MODELS_DIR, subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, model_name)

        if os.path.exists(dest_path):
            print(f"   ✅ [OK] {model_name} already present in {dest_dir}")
            continue

        if model_name in found_files:
            src_path = found_files[model_name]
            try:
                os.symlink(src_path, dest_path)
                print(f"   ✅ [LINKED] {model_name} -> {dest_path}")
            except Exception as e:
                print(f"   ❌ Failed to symlink {model_name}: {e}")
        else:
            # Fallback heuristic search
            matched = False
            for fname, fpath in found_files.items():
                if model_name.lower().split(".")[0] in fname.lower():
                    try:
                        os.symlink(fpath, dest_path)
                        print(f"   ✅ [HEURISTIC LINKED] {fname} as {dest_path}")
                        matched = True
                        break
                    except Exception as e:
                        print(f"   ❌ Failed heuristic link: {e}")
            if not matched:
                print(f"   ⚠️ [MISSING] {model_name} (expected in {dest_dir})")

    # 4. Summary of ComfyUI model directories
    print("\n📊 Final ComfyUI models directory status:")
    if os.path.exists(COMFY_MODELS_DIR):
        for sub in sorted(os.listdir(COMFY_MODELS_DIR)):
            sub_path = os.path.join(COMFY_MODELS_DIR, sub)
            if os.path.isdir(sub_path):
                models_in_sub = os.listdir(sub_path)
                if models_in_sub:
                    print(f"   📁 {sub}: {models_in_sub}")
                else:
                    print(f"   📁 {sub}: (empty)")
    print("=" * 60)

if __name__ == "__main__":
    scan_and_link()
