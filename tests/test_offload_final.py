# ==============================================================================
# The workflow actually POSTed to ComfyUI /prompt (real handler.queue_prompt, urlopen mocked):
# 480p on a >= 28 GB GPU always queues blocks_to_swap 0 / force_offload False / main_device,
# whatever the workflow default (38 / True), the client, manual mode or OFFLOAD_TIERS say,
# with the guard on and with WORKER_GUARD_ENABLED=0. Only the emergency OOM retry may change it.
# Also: "[OFFLOAD] final:" log line, the fail-safe error, 720p unchanged, opt-in vae_tiling.
# ==============================================================================
import base64
import copy
import json
import logging
import urllib.request
from pathlib import Path

import pytest

import handler
from worker_guard import comfy_api, job_guard, offload
from test_handler_video_regression import VIDEO_BYTES, FakeWs, run_unguarded, strip_task_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
GB = 1024 ** 3
RTX5090 = {"devices": [{"type": "cuda", "name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync", "index": 0,
                        "vram_total": int(31.4 * GB), "vram_free": int(30.5 * GB)}]}
CLIENT_38 = {"blocks_to_swap": 38, "force_offload": True, "load_device": "offload_device"}
WORKFLOWS = {"image": "I2V_single.json", "video": "V2V_single.json"}


class FakeResponse:
    def read(self):
        return b'{"prompt_id": "p1"}'


@pytest.fixture
def posted(tmp_path, monkeypatch):
    """Real queue_prompt; records the workflow POSTed to /prompt."""
    video = tmp_path / "out" / "InfiniteTalk_00001.mp4"
    video.parent.mkdir()
    video.write_bytes(VIDEO_BYTES)
    sent = []

    def fake_urlopen(req, timeout=None):
        if isinstance(req, urllib.request.Request) and req.full_url.endswith("/prompt"):
            sent.append(json.loads(req.data)["prompt"])
            return FakeResponse()
        return object()  # handler's HTTP readiness probe

    monkeypatch.setenv("TORCH_COMPILE", "0")
    monkeypatch.setattr(comfy_api, "system_stats", lambda timeout=10: RTX5090)
    monkeypatch.setattr(handler, "COMFY_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setattr(handler, "get_workflow_path", lambda it, pc: str(REPO_ROOT / WORKFLOWS[it]))
    monkeypatch.setattr(handler, "load_workflow", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(handler, "get_audio_duration", lambda path: 4.0)
    monkeypatch.setattr(handler, "get_history", lambda pid: {pid: {"outputs": {"131": {"gifs": [{"fullpath": str(video)}]}}}})
    monkeypatch.setattr(handler.websocket, "WebSocket", FakeWs)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


def job(input_type="image", width=854, height=480, **extra):
    payload = {
        "wav_base64": base64.b64encode(b"RIFF fake wav").decode(),
        "prompt": "A person talking naturally",
        "width": width,
        "height": height,
        "input_type": input_type,
    }
    if input_type == "image":
        payload["image_base64"] = base64.b64encode(b"\x89PNG fake image").decode()
    else:
        payload["video_base64"] = base64.b64encode(b"fake mp4").decode()
    payload.update(extra)
    return {"input": payload}


def offload_nodes(prompt):
    return (prompt["134"]["inputs"]["blocks_to_swap"], prompt["134"]["inputs"]["offload_img_emb"],
            prompt["134"]["inputs"]["offload_txt_emb"], prompt["128"]["inputs"]["force_offload"],
            prompt["122"]["inputs"]["load_device"])


FULL_GPU_NODES = (0, False, False, False, "main_device")


@pytest.mark.parametrize("workflow", ["image", "video"])
def test_workflow_defaults_are_38_and_force_offload(workflow):
    w = json.loads((REPO_ROOT / WORKFLOWS[workflow]).read_text(encoding="utf-8"))
    assert offload_nodes(w) == (38, True, True, True, "offload_device")


# ---------------------------------------------------------------- the requested test: workflow 38 + client 38 -> 0
@pytest.mark.parametrize("workflow", ["image", "video"])
@pytest.mark.parametrize("size", [(854, 480), (720, 480), (512, 512)])
@pytest.mark.parametrize("variant", ["auto", "manual_allowed", "tiers_38"])
def test_480p_queues_0_with_workflow_38_and_client_38(posted, monkeypatch, caplog, workflow, size, variant):
    extra = dict(CLIENT_38)
    if variant == "manual_allowed":
        monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
        extra["offload_mode"] = "manual"
    elif variant == "tiers_38":
        monkeypatch.setenv("OFFLOAD_TIERS", json.dumps([{"min_vram_gb": 28,
                           "480p": {"blocks_to_swap": 38, "force_offload": True},
                           "720p": {"blocks_to_swap": 38, "force_offload": True}}]))
    caplog.set_level(logging.INFO)
    res = handler.handler(job(workflow, *size, **extra))
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert len(posted) == 1
    assert offload_nodes(posted[0]) == FULL_GPU_NODES
    assert ("[OFFLOAD] final: blocks_to_swap 0, force_offload False, node 134 {'blocks_to_swap': 0, "
            "'offload_img_emb': False, 'offload_txt_emb': False}, load_device main_device") in caplog.messages
    assert not any("Fixed before queueing" in m for m in caplog.messages)  # the decision was already right


@pytest.mark.parametrize("workflow", ["image", "video"])
def test_480p_queues_0_with_guard_disabled(posted, monkeypatch, caplog, workflow):
    caplog.set_level(logging.INFO)
    res = run_unguarded(monkeypatch, job(workflow, **CLIENT_38))  # handler uses the client's 38 / True
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert offload_nodes(posted[0]) == FULL_GPU_NODES
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == ["[OFFLOAD] final workflow has blocks_to_swap 38, force_offload True, load_device offload_device "
                      "(GPU 31.4 GB, 854x480, oom_retry no); expected blocks_to_swap 0, force_offload False. "
                      "Fixed before queueing"]
    assert any(m.startswith("[OFFLOAD] final: blocks_to_swap 0, force_offload False") for m in caplog.messages)


# ---------------------------------------------------------------- v24 fast480: 480p rule, proven by "[OFFLOAD] final"
FINAL_FULL_GPU = ("[OFFLOAD] final: blocks_to_swap 0, force_offload False, node 134 {'blocks_to_swap': 0, "
                  "'offload_img_emb': False, 'offload_txt_emb': False}, load_device main_device")


@pytest.mark.parametrize("aspect, size", [(None, (848, 480)), ("16:9", (848, 480)), ("9:16", (480, 848))])
@pytest.mark.parametrize("guard", ["on", "off"])
def test_fast480_queues_full_gpu(posted, monkeypatch, caplog, aspect, size, guard):
    extra = {"mode": "fast480", "upscale": "none", **CLIENT_38}  # v26: upscale covered in test_upscale_job.py
    if aspect:
        extra["aspect"] = aspect
    caplog.set_level(logging.INFO)
    payload = job(width=1280, height=720, **extra)  # the client's size is replaced
    res = handler.handler(payload) if guard == "on" else run_unguarded(monkeypatch, payload)
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    prompt = posted[0]
    assert (prompt["245"]["inputs"]["value"], prompt["246"]["inputs"]["value"]) == size
    assert offload_nodes(prompt) == FULL_GPU_NODES
    assert FINAL_FULL_GPU in caplog.messages
    if guard == "on":
        assert not any("Fixed before queueing" in m for m in caplog.messages)
        assert any(m.startswith(f"[JOB] fast480 done: res {size[0]}x{size[1]}, gen ") for m in caplog.messages)


def test_normal_480p_final_line(posted, caplog):
    caplog.set_level(logging.INFO)
    handler.handler(job(width=854, height=480))
    assert FINAL_FULL_GPU in caplog.messages
    assert any(m.startswith("[JOB] standard done: res 854x480, ") for m in caplog.messages)


@pytest.mark.parametrize("extra", [{"input_type": "video", "mode": "fast480"}, {"mode": "fast480", "person_count": "multi"},
                                   {"mode": "nope"}])
@pytest.mark.parametrize("guard", ["on", "off"])
def test_invalid_mode_is_rejected_before_queueing(posted, monkeypatch, extra, guard):
    payload = job(**extra)
    res = handler.handler(payload) if guard == "on" else run_unguarded(monkeypatch, payload)
    assert res["error"].startswith("[MODE] ")
    assert posted == []


def test_guard_disabled_and_vram_unknown_still_queues_0(posted, monkeypatch, caplog):
    def down(timeout=10):
        raise ConnectionError("refused")
    monkeypatch.setattr(comfy_api, "system_stats", down)
    caplog.set_level(logging.INFO)
    run_unguarded(monkeypatch, job(**CLIENT_38))
    assert offload_nodes(posted[0]) == FULL_GPU_NODES
    assert any("assuming >= 28 GB" in m for m in caplog.messages)


def test_fail_safe_fixes_a_wrong_decision_with_guard_on(posted, monkeypatch, caplog):
    monkeypatch.setattr(offload, "apply", lambda job_input, decision: job_input)  # simulated bug: client values leak
    caplog.set_level(logging.INFO)
    handler.handler(job(**CLIENT_38))
    assert offload_nodes(posted[0]) == FULL_GPU_NODES
    assert any(r.levelno == logging.ERROR and "Fixed before queueing" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- 720p, OOM retry, larger GPUs
def test_720p_on_32gb_keeps_the_tier_values(posted, caplog):
    caplog.set_level(logging.INFO)
    handler.handler(job(width=1280, height=720, **CLIENT_38))
    assert offload_nodes(posted[0]) == (25, True, True, True, "offload_device")
    assert not any("Fixed before queueing" in m for m in caplog.messages)


def test_720p_with_guard_disabled_is_not_touched(posted, monkeypatch):
    run_unguarded(monkeypatch, job(width=1280, height=720, **CLIENT_38))
    assert offload_nodes(posted[0]) == (38, True, True, True, "offload_device")


def test_fail_safe_fixes_720p_mismatch_with_guard_on(posted, monkeypatch, caplog):
    monkeypatch.setattr(offload, "apply", lambda job_input, decision: job_input)
    caplog.set_level(logging.INFO)
    handler.handler(job(width=1280, height=720, **CLIENT_38))
    assert offload_nodes(posted[0]) == (25, True, True, True, "offload_device")
    assert any("expected blocks_to_swap 25, force_offload True. Fixed before queueing" in m for m in caplog.messages)


def prompt_with(blocks, force, width=854, height=480):
    w = json.loads((REPO_ROOT / "I2V_single.json").read_text(encoding="utf-8"))
    w["245"]["inputs"]["value"], w["246"]["inputs"]["value"] = width, height
    offload._set_offload(w, blocks, force)
    return w


def test_oom_retry_may_swap_on_480p(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    p = prompt_with(6, False)
    assert offload.enforce_final(p, 31.4, (6, False), oom_retry=True) == (6, False)
    assert offload_nodes(p) == (6, True, True, False, "offload_device")
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


def test_oom_retry_through_the_guard_context(monkeypatch, caplog):
    ctx = job_guard.JobContext(job_guard.Settings.from_env(), "video")
    ctx.offload = offload.Decision("auto", 6, False, 31.4, 854, 480, "480p")
    ctx.oom_retry = True
    monkeypatch.setattr(job_guard, "_current", ctx)
    p = prompt_with(6, False)
    assert job_guard.final_offload_check(p) == (6, False)
    ctx.oom_retry = False  # the same 6 blocks outside the retry are wrong
    assert job_guard.final_offload_check(p) == (0, False)
    assert offload_nodes(p) == FULL_GPU_NODES


@pytest.mark.parametrize("vram_gb, expected", [(27.9, (38, True)), (28.0, (0, False)), (79.2, (0, False))])
def test_hard_rule_threshold(vram_gb, expected):
    assert offload.enforce_final(prompt_with(38, True), vram_gb) == expected


def test_480p_with_main_device_mismatch_is_fixed():
    p = prompt_with(0, False)
    p["122"]["inputs"]["load_device"] = "offload_device"
    offload.enforce_final(p, 31.4, (0, False))
    assert offload_nodes(p) == FULL_GPU_NODES


# ---------------------------------------------------------------- opt-in VAE tiling (TESTING.md step 2c)
@pytest.mark.parametrize("workflow, nodes", [
    ("image", {"192": "tiled_vae", "130": "enable_vae_tiling"}),
    ("video", {"192": "tiled_vae", "130": "enable_vae_tiling", "229": "enable_vae_tiling"}),
])
def test_vae_tiling_only_when_requested(posted, workflow, nodes):
    handler.handler(job(workflow))
    handler.handler(job(workflow, vae_tiling=True))
    off, on = posted
    for node_id, name in nodes.items():
        assert off[node_id]["inputs"][name] is False and on[node_id]["inputs"][name] is True
    strip = copy.deepcopy(on)
    for node_id, name in nodes.items():
        strip[node_id]["inputs"][name] = False
    assert strip_task_ids(strip) == strip_task_ids(off)  # nothing else changes
