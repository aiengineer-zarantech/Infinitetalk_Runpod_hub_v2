"""Dockerfile.v26 build check (CPU, no GPU, no real models): every SeedVR2 segment prompt the upscale stage
can send (both models, first try + the 3 OOM retries, 16:9 and 9:16) is accepted by the installed ComfyUI's
own /prompt validation, and every node it uses is registered.

Runs ComfyUI from a throwaway copy of /ComfyUI (with empty placeholder model files so the loader lists
contain the real file names) on port 8199, then deletes the copy. Nothing is written to the image.
Usage: python smoke_v26.py <worker root>
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

root = sys.argv[1]
sys.path.insert(0, root)
from worker_guard import upscale  # noqa: E402

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
        return json.loads(data) if data else None


work = tempfile.mkdtemp(prefix="smoke_v26_")
comfy = os.path.join(work, "ComfyUI")
proc = None
try:
    shutil.copytree("/ComfyUI", comfy, symlinks=True)
    os.makedirs(os.path.join(comfy, "models", "diffusion_models"), exist_ok=True)
    for name in upscale.MODEL_FILES.values():
        open(os.path.join(comfy, "models", "diffusion_models", name), "wb").close()
    os.makedirs(os.path.join(comfy, "models", "vae"), exist_ok=True)
    open(os.path.join(comfy, "models", "vae", upscale.VAE_FILE), "wb").close()
    clip = os.path.join(work, "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=848x480:rate=25",
                    "-frames:v", "20", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)

    log = open(os.path.join(work, "comfy.log"), "w")
    env = {**os.environ, "VRAM_STAGES": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PROFILE", None)
    proc = subprocess.Popen(["/usr/bin/python3.10", "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(PORT)],
                            cwd=comfy, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(300):
        try:
            get("/system_stats")
            break
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"ComfyUI exited ({proc.returncode}):\n" + open(log.name).read()[-3000:])
            time.sleep(1)
    else:
        raise SystemExit("ComfyUI did not start within 300 s:\n" + open(log.name).read()[-3000:])

    template = upscale.load_template(root)
    classes = sorted({n["class_type"] for n in template.values()})
    for node_class in classes:
        info = get(f"/object_info/{node_class}")
        assert node_class in info, f"{node_class} is not registered in ComfyUI"
    for node_class, input_name in upscale.LOADERS.items():
        spec = get(f"/object_info/{node_class}")[node_class]["input"]["required"][input_name]
        options = spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
        wanted = list(upscale.MODEL_FILES.values()) if node_class == "UNETLoader" else [upscale.VAE_FILE]
        assert all(w in options for w in wanted), f"{node_class} does not list {wanted}"
    print(f"OK: nodes registered: {', '.join(classes)}")

    probe = upscale.probe(clip)
    segment = upscale.plan_segments(probe.frames, 9)[1]
    checked = 0
    for target in (upscale.LANDSCAPE, upscale.PORTRAIT):
        for model in upscale.MODEL_FILES.values():
            for attempt in (upscale.FIRST_ATTEMPT,) + upscale.OOM_RETRIES:
                prompt = upscale.build_prompt(template, clip, segment, target, model, attempt, probe.fps_value,
                                              "lab", "smoke")
                try:
                    answer = post("/prompt", {"prompt": prompt, "client_id": "smoke"})
                except urllib.error.HTTPError as e:
                    raise SystemExit(f"REJECTED {model} {target} {attempt}: {e.read().decode()[:2000]}")
                assert not answer.get("node_errors"), answer
                post("/queue", {"clear": True})
                post("/interrupt", {})
                checked += 1
    print(f"OK: ComfyUI /prompt accepted all {checked} upscale prompt variants")
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    shutil.rmtree(work, ignore_errors=True)
