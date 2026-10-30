"""Runs jobs through the REAL handler.py of a worker tree with ComfyUI mocked and prints, as JSON,
every workflow POSTed to /prompt plus each job's result (input task ids normalised).
Used by test_v23_prompt_identity.py to compare v1.0-stable with the current tree.
Usage: python prompt_driver.py <worker root> <jobs.json> <out.json>
"""
import base64
import json
import os
import re
import sys
import tempfile
import urllib.request
from unittest.mock import MagicMock

root, jobs_path, out_path = sys.argv[1:4]
sys.path.insert(0, root)
os.chdir(root)
for name in ("runpod", "runpod.serverless", "websocket", "soundfile", "librosa", "librosa.effects"):
    sys.modules[name] = MagicMock()
tmp = tempfile.mkdtemp()
os.environ.update(WORKFLOW_DIR=root, WORKER_GPU_INFO=os.path.join(tmp, "none.json"))

import handler  # noqa: E402
import setup_models  # noqa: E402
from worker_guard import comfy_api  # noqa: E402

GB = 1024 ** 3
MULTI = "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"
STATS = {"devices": [{"type": "cuda", "name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync", "index": 0,
                      "vram_total": int(31.4 * GB), "vram_free": int(30.5 * GB)}]}
VIDEO = os.path.join(tmp, "out.mp4")
with open(VIDEO, "wb") as f:
    f.write(b"\x00\x00\x00\x18ftypmp42 fake mp4 payload")

comfy_api.system_stats = lambda timeout=10: STATS
comfy_api.free = lambda **kwargs: None
comfy_api.object_info = lambda node_class, timeout=30: {node_class: {"input": {"required": {"model": [[MULTI], {}]}}}}
setup_models.ensure_model = lambda name, subfolder=None: {
    "model": name, "copied": False, "seconds": 0.0, "dest_path": "/d", "local_path": "/l"}

sent = []


class FakeResponse:
    def read(self):
        return b'{"prompt_id": "p1"}'


def fake_urlopen(req, timeout=None):
    if isinstance(req, urllib.request.Request) and req.full_url.endswith("/prompt"):
        sent.append(json.loads(req.data)["prompt"])
        return FakeResponse()
    raise ConnectionError("not ComfyUI")  # readiness probe below, /worker/vram (v24) and the like


class FakeWs:
    def __init__(self):
        self.messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}}),
            json.dumps({"type": "executing", "data": {"node": "128", "prompt_id": "p1"}}),
            *[json.dumps({"type": "progress", "data": {"value": v, "max": 12, "node": "128", "prompt_id": "p1"}})
              for v in range(1, 13)],
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
        ]

    def connect(self, url):
        pass

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        pass


original_paths = handler.get_workflow_path
handler.get_workflow_path = lambda it, pc: os.path.join(root, os.path.basename(original_paths(it, pc)))
handler.load_workflow = lambda path: json.load(open(path, encoding="utf-8"))
handler.get_audio_duration = lambda path: 4.0
handler.get_history = lambda pid: {pid: {"outputs": {"131": {"gifs": [{"fullpath": VIDEO}]}}}}
handler.COMFY_INPUT_DIR = os.path.join(tmp, "input")
handler.websocket.WebSocket = FakeWs
handler.time.sleep = lambda s: None
# The readiness probe must succeed: only it calls urlopen with a plain URL string and timeout=5
urllib.request.urlopen = lambda req, timeout=None: (
    object() if isinstance(req, str) and req.endswith(":8188/") else fake_urlopen(req, timeout))

results = []
with open(jobs_path, encoding="utf-8") as f:
    jobs = json.load(f)
for env, job_input in jobs:
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        results.append(handler.handler({"input": job_input}))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

text = json.dumps({"prompts": sent, "results": results}, sort_keys=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(re.sub(r"task_[0-9a-f-]{36}", "task_X", text))
