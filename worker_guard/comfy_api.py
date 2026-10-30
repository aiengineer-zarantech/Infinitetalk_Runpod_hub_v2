"""Small ComfyUI HTTP helpers (urllib only, explicit timeouts, no secrets involved)."""
import json
import os
import urllib.parse
import urllib.request

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))


def base_url():
    return f"http://{SERVER_ADDRESS}:{COMFY_PORT}"


def _get_json(path, timeout):
    with urllib.request.urlopen(base_url() + path, timeout=timeout) as response:
        return json.loads(response.read())


def system_stats(timeout=10):
    return _get_json("/system_stats", timeout)


def first_device(stats):
    """Returns the first device from /system_stats as a flat dict, or None if none is reported."""
    devices = (stats or {}).get("devices") or []
    if not devices:
        return None
    device = devices[0]
    total = int(device.get("vram_total") or 0)
    free = int(device.get("vram_free") or 0)
    return {
        "type": str(device.get("type", "")),
        "name": str(device.get("name", "")),
        "index": device.get("index"),
        "vram_total": total,
        "vram_free": free,
        "vram_used": max(0, total - free),
    }


def interrupt(prompt_id=None, timeout=10):
    """POST /interrupt. With prompt_id, newer ComfyUI only interrupts that prompt."""
    body = json.dumps({"prompt_id": prompt_id} if prompt_id else {}).encode("utf-8")
    req = urllib.request.Request(base_url() + "/interrupt", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        response.read()


def free(unload_models=True, free_memory=True, timeout=10):
    """POST /free. ComfyUI applies it asynchronously in its prompt worker loop."""
    body = json.dumps({"unload_models": unload_models, "free_memory": free_memory}).encode("utf-8")
    req = urllib.request.Request(base_url() + "/free", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        response.read()


def queue(prompt, client_id, timeout=30):
    """POST /prompt. Returns ComfyUI's answer ({"prompt_id": ..., "node_errors": {...}})."""
    body = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(base_url() + "/prompt", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def history(prompt_id, timeout=10):
    """GET /history/<prompt_id>: {} while the prompt is still queued or running."""
    return _get_json(f"/history/{urllib.parse.quote(prompt_id)}", timeout)


def vram_stages(timeout=5):
    """GET /worker/vram (zz_worker_vram package): per-stage VRAM peaks of the last prompt."""
    return _get_json("/worker/vram", timeout)


def object_info(node_class, timeout=30):
    return _get_json(f"/object_info/{urllib.parse.quote(node_class)}", timeout)


def list_models(folder, timeout=30):
    return _get_json(f"/models/{urllib.parse.quote(folder)}", timeout)
