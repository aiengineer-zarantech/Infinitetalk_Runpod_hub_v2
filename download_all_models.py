import requests
import os
import time
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT_ID = "ghuoueuyp8iu7g"
API_KEY = os.environ["RUNPOD_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

MODELS_TO_DOWNLOAD = [
    {
        "name": "LoRA (lightx2v)",
        "file": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "dir": "/runpod-volume/models/loras",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "size_expected": "704 MB"
    },
    {
        "name": "CLIP Vision (clip_vision_h)",
        "file": "clip_vision_h.safetensors",
        "dir": "/runpod-volume/models/clip_vision",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        "size_expected": "1.2 GB"
    },
    {
        "name": "InfiniteTalk Single",
        "file": "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors",
        "dir": "/runpod-volume/models/diffusion_models",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors",
        "size_expected": "2.6 GB"
    },
    {
        "name": "InfiniteTalk Multi",
        "file": "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors",
        "dir": "/runpod-volume/models/diffusion_models",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors",
        "size_expected": "2.6 GB"
    },
    {
        "name": "Text Encoder (umt5-xxl)",
        "file": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "dir": "/runpod-volume/models/text_encoders",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "size_expected": "6.4 GB"
    },
    {
        "name": "Base I2V Diffusion (14B 480P)",
        "file": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "dir": "/runpod-volume/models/diffusion_models",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "size_expected": "16.2 GB"
    }
]

def run_worker_cmd(cmd, max_wait_sec=900):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    payload = {"input": {"cmd": cmd}}
    
    res = requests.post(url, json=payload, headers=headers, timeout=15)
    res.raise_for_status()
    job_id = res.json()["id"]
    print(f"   [Job {job_id}] Started...", flush=True)
    
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    start_time = time.time()
    
    while time.time() - start_time < max_wait_sec:
        s_res = requests.get(status_url, headers=headers, timeout=15)
        if s_res.status_code == 200:
            data = s_res.json()
            status = data.get("status")
            if status == "COMPLETED":
                output = data.get("output", {})
                return True, output.get("output", "")
            elif status in ["FAILED", "CANCELLED"]:
                return False, str(data)
        time.sleep(5)
    return False, "Timed out waiting for worker command"

def check_existing_files():
    print("🔍 Checking existing models in /runpod-volume/models...")
    ok, out = run_worker_cmd("find /runpod-volume/models -type f -exec ls -lh {} \\;")
    if ok:
        print(out)
    else:
        print(f"Error checking models: {out}")

def download_missing():
    print("\n" + "="*70)
    print("🚀 AUTOMATED MODEL DOWNLOAD TO RUNPOD GLOBAL VOLUME (/runpod-volume)")
    print("="*70)
    
    for idx, item in enumerate(MODELS_TO_DOWNLOAD, 1):
        target_path = f"{item['dir']}/{item['file']}"
        print(f"\n[{idx}/{len(MODELS_TO_DOWNLOAD)}] Checking {item['name']} ({item['size_expected']})...")
        
        # Check if already downloaded
        check_cmd = f"test -f {target_path} && ls -lh {target_path}"
        ok, out = run_worker_cmd(check_cmd, max_wait_sec=30)
        if ok and item['file'] in out:
            print(f"   ✅ Already present: {out.strip()}")
            continue
            
        print(f"   ⬇️ Downloading from {item['url']} -> {target_path}...")
        t0 = time.time()
        dl_cmd = f"mkdir -p {item['dir']} && wget -c -q '{item['url']}' -O {target_path} && sync && ls -lh {target_path}"
        ok, out = run_worker_cmd(dl_cmd, max_wait_sec=900)
        duration = time.time() - t0
        
        if ok and item['file'] in out:
            print(f"   🎉 SUCCESS in {duration:.1f}s! {out.strip()}")
        else:
            print(f"   ❌ FAILED after {duration:.1f}s! Error: {out}")
            sys.exit(1)

    print("\n" + "="*70)
    print("✅ ALL REQUIRED MODELS ARE DOWNLOADED AND SYNCED TO GLOBAL VOLUME!")
    print("="*70)
    check_existing_files()

if __name__ == "__main__":
    download_missing()
