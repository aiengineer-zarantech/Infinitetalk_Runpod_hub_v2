"""Dockerfile.v27 build check (CPU, no GPU, no real models), checked against the installed versions:
- WanVideoModelLoader lists both ATTENTION_MODE values ("sageattn", "sdpa") in its attention_mode input, and
  every workflow's node 122 already uses "sageattn" (so default jobs are not changed);
- /system_stats reports argv (used for the [COMFY] launch flags line);
- the sageattn self-test returns a "failed: ..." status instead of raising when there is no GPU;
- the installed sageattention / triton versions are the expected ones (nothing was installed).

Runs ComfyUI from a throwaway copy of /ComfyUI on port 8199, then deletes the copy. Nothing is written to the image.
Usage: python smoke_v27.py <worker root>
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import torch

root = sys.argv[1]
sys.path.insert(0, root)
from worker_guard import attention  # noqa: E402

PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"
WORKFLOWS = ("I2V_single.json", "I2V_multi.json", "V2V_single.json", "V2V_multi.json",
             "I2V_single_compiled.json", "I2V_multi_compiled.json")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


status = attention.self_test(torch)
assert status.startswith("failed: "), status
print(f"OK: sageattn self-test without a GPU returns {status[:120]!r}")
versions = attention.package_versions()
assert versions == {"sageattention": "2.2.0", "triton": "3.5.1"}, versions
print(f"OK: versions {versions}")

for name in WORKFLOWS:
    with open(os.path.join(root, name), encoding="utf-8") as f:
        prompt = json.load(f)
    loaders = [n for n in prompt.values() if n.get("class_type") == attention.LOADER_CLASS]
    assert loaders and all(n["inputs"]["attention_mode"] == attention.DEFAULT_MODE for n in loaders), name
    assert attention.apply(prompt, attention.DEFAULT_MODE) == [], name
print(f"OK: all {len(WORKFLOWS)} workflows use {attention.DEFAULT_MODE}; default jobs are not changed")

work = tempfile.mkdtemp(prefix="smoke_v27_")
comfy = os.path.join(work, "ComfyUI")
proc = None
try:
    shutil.copytree("/ComfyUI", comfy, symlinks=True)
    log = open(os.path.join(work, "comfy.log"), "w")
    env = {**os.environ, "VRAM_STAGES": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PROFILE", None)
    proc = subprocess.Popen(["/usr/bin/python3.10", "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(PORT)],
                            cwd=comfy, stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(300):
        try:
            stats = get("/system_stats")
            break
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"ComfyUI exited ({proc.returncode}):\n" + open(log.name).read()[-3000:])
            time.sleep(1)
    else:
        raise SystemExit("ComfyUI did not start within 300 s:\n" + open(log.name).read()[-3000:])

    argv = stats.get("system", {}).get("argv")
    assert isinstance(argv, list) and "--cpu" in argv, stats.get("system")
    print(f"OK: /system_stats argv {argv}")

    inputs = get(f"/object_info/{attention.LOADER_CLASS}")[attention.LOADER_CLASS]["input"]
    # wrapper 088128b lists attention_mode under "optional"
    spec = {**inputs.get("required", {}), **inputs.get("optional", {})}["attention_mode"]
    options = spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
    assert all(m in options for m in attention.MODES), options
    print(f"OK: {attention.LOADER_CLASS}.attention_mode options include {', '.join(attention.MODES)} ({options})")
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    shutil.rmtree(work, ignore_errors=True)
