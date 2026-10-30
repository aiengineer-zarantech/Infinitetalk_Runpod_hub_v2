import urllib.request

urls = [
    ('clip_vision', 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors'),
    ('lora', 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'),
    ('single_talk', 'https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors'),
    ('multi_talk', 'https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors'),
    ('text_encoder', 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors'),
    ('base_i2v', 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors')
]

for name, url in urls:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        # Hugging Face resolve redirects with 302, urllib handles redirects
        with urllib.request.urlopen(req, timeout=10) as resp:
            size_mb = int(resp.headers.get("Content-Length", 0)) / (1024 * 1024)
            print(f"{name}: OK - {size_mb:.2f} MB")
    except Exception as e:
        print(f"{name}: FAILED - {e}")
