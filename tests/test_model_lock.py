"""MODEL_LOCK: load once, reuse on every job, no reload, no OOM escalation (fake ComfyUI, real I2V workflow)."""
import copy
import json
import logging
import os

import pytest

from worker_guard import comfy_api, job_guard, model_lock

GB = 1024 ** 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = json.load(open(os.path.join(ROOT, "I2V_single.json")))
CFG = model_lock.config()


# ---------------------------------------------------------------------------------------------- unit
def test_config_values_and_hash_are_stable():
    assert CFG["blocks_to_swap"] == 0 and CFG["load_device"] == "main_device" and CFG["torch_compile"] is True
    assert CFG["vram"]["headroom_gb"] == 2.5
    assert model_lock.config_hash() == model_lock.config_hash(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other["blocks_to_swap"] = 10
    assert model_lock.config_hash(other) != model_lock.config_hash()


def test_sanitize_strips_model_fields_and_locks_offload():
    job = {"prompt": "hi", "image_url": "u", "wav_base64": "x", "width": 848, "height": 480, "max_frame": 300,
           "blocks_to_swap": 25, "force_offload": True, "torch_compile": False, "fp8_fast": True, "benchmark": True,
           "attention_mode": "sdpa", "vae_tiling": True, "upscale": "720p", "negative_prompt": "n"}
    out, ignored, err = model_lock.sanitize(job)
    assert err is None
    assert {k: out[k] for k in ("blocks_to_swap", "force_offload", "load_device", "base_precision")} == \
        {"blocks_to_swap": 0, "force_offload": False, "load_device": "main_device", "base_precision": "bf16"}
    for key in ("torch_compile", "fp8_fast", "benchmark", "attention_mode", "vae_tiling", "upscale"):
        assert key not in out
    assert {k: out[k] for k in ("prompt", "image_url", "wav_base64", "max_frame", "negative_prompt")} == \
        {"prompt": "hi", "image_url": "u", "wav_base64": "x", "max_frame": 300, "negative_prompt": "n"}
    assert "blocks_to_swap=25" in ignored and "benchmark=True" in ignored


def test_sanitize_values_equal_to_the_lock_are_not_reported():
    _, ignored, _ = model_lock.sanitize({"blocks_to_swap": 0, "force_offload": False})
    assert ignored == []


@pytest.mark.parametrize("job, words", [
    ({"person_count": "multi"}, "multi-person"),
    ({"input_type": "video"}, "video input"),
    ({"width": 1280, "height": 720}, "larger than the locked"),
])
def test_sanitize_rejects_what_the_lock_cannot_run(job, words):
    out, _, err = model_lock.sanitize(job)
    assert out is None and words in err


def test_enforce_fixes_the_loader_chain_and_is_idempotent():
    prompt = copy.deepcopy(WORKFLOW)            # the raw workflow: blocks_to_swap 38, offload_device
    fixes = model_lock.enforce(prompt)
    assert any(f.startswith("134.blocks_to_swap") for f in fixes) and any("122.load_device" in f for f in fixes)
    assert prompt["134"]["inputs"]["blocks_to_swap"] == 0 and prompt["128"]["inputs"]["force_offload"] is False
    assert model_lock.enforce(prompt) == []


def test_predict_peak_grows_with_frames_and_tiling_caps_it():
    short, long_ = model_lock.predict_peak_gb(831, False), model_lock.predict_peak_gb(3081, False)
    assert 22.4 < short < long_
    assert model_lock.predict_peak_gb(3081, True) == pytest.approx(22.4 + CFG["vram"]["tiled_decode_gb"])


def dev(used_gb, total_gb=31.4):
    return {"vram_total": int(total_gb * GB), "vram_free": int((total_gb - used_gb) * GB)}


def prompt_with_frames(n):
    p = copy.deepcopy(WORKFLOW)
    p["270"]["inputs"]["value"] = n
    return p


def test_preflight_ok_without_any_action():
    p = prompt_with_frames(831)                  # 30 s
    rep = model_lock.preflight(p, dev(20.0), int(20.0 * GB), lambda: pytest.fail("no clear"), None)
    assert rep["ok"] and not rep["cleared"] and not rep["tiled"] and rep["headroom_gb"] >= 2.5
    assert p["130"]["inputs"]["enable_vae_tiling"] is False


def test_preflight_two_minute_job_gets_tiled_decode():
    p = prompt_with_frames(3081)                 # 2 min
    rep = model_lock.preflight(p, dev(20.0), int(20.0 * GB), lambda: False, None)
    assert rep["ok"] and rep["tiled"] and p["130"]["inputs"]["enable_vae_tiling"] is True


def test_preflight_clears_non_model_memory_first():
    calls = []
    p = prompt_with_frames(831)
    rep = model_lock.preflight(p, dev(27.0), int(20.0 * GB), lambda: calls.append(1) or True, lambda: dev(20.5))
    assert calls == [1] and rep["cleared"] and rep["ok"] and not rep["tiled"]


def test_preflight_fails_fast_when_nothing_helps():
    p = prompt_with_frames(3081)
    rep = model_lock.preflight(p, dev(30.0), int(20.0 * GB), lambda: True, lambda: dev(30.0))
    assert not rep["ok"] and "Failing fast" in rep["message"]


def test_resident_state_reuse_mismatch_and_oom():
    st = model_lock.ResidentState()
    assert st.check({"a": 1}) == ("cold", None)
    st.after_job({"a": 1}, ok=True, oom=False, idle_bytes=20 * GB, status="cold")
    assert st.check({"a": 1}) == ("reused", None)
    status, msg = st.check({"a": 2})
    assert status == "mismatch" and "refusing to load a second copy" in msg
    st.after_job({"a": 1}, ok=False, oom=True, idle_bytes=0, status="reused")
    assert st.check({"a": 1}) == ("cold", None) and st.idle_bytes == 0


# ------------------------------------------------------------------------------------------ integration
class Ws:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv(self):
        return json.dumps(self.messages.pop(0))


def messages(cached):
    return [{"type": "execution_start", "data": {"prompt_id": "p1"}},
            {"type": "execution_cached", "data": {"nodes": cached, "prompt_id": "p1"}},
            {"type": "executing", "data": {"node": "128", "prompt_id": "p1"}},
            {"type": "executing", "data": {"node": None, "prompt_id": "p1"}}]


@pytest.fixture
def worker(monkeypatch):
    """A locked worker with a fake ComfyUI. world.used_gb is what /system_stats reports."""
    monkeypatch.setenv("MODEL_LOCK", "1")
    monkeypatch.setattr(model_lock, "STATE", model_lock.ResidentState())
    world = {"used_gb": 1.0, "clears": 0, "queued": [], "cached": [], "result": None, "frames": 831}

    def system_stats(timeout=10):
        total = 31.4 * GB
        return {"devices": [{"type": "cuda", "name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync", "index": 0,
                             "vram_total": int(total), "vram_free": int(total - world["used_gb"] * GB)}]}

    monkeypatch.setattr(comfy_api, "system_stats", system_stats)
    monkeypatch.setattr(comfy_api, "vram_stages", lambda timeout=5: {"stages": {"sampler": int(23 * GB)}, "total": int(31.4 * GB)})
    monkeypatch.setattr(comfy_api, "clear_cache", lambda timeout=30: world.__setitem__("clears", world["clears"] + 1) or {})
    monkeypatch.setattr(comfy_api, "interrupt", lambda prompt_id=None, timeout=10: None)

    def get_videos(ws, prompt, input_type="image", person_count="single"):
        for _ in range(4):
            ws.recv()
        world["queued"].append(copy.deepcopy(prompt))
        world["used_gb"] = 20.0                       # model resident after the job
        return {"131": ["/out.mp4"]}

    def inner(job):
        """What handler.py does with the job input before get_videos."""
        ji = job["input"]
        p = copy.deepcopy(WORKFLOW)
        p["270"]["inputs"]["value"] = world["frames"]
        blocks = int(ji.get("blocks_to_swap", 0))
        p["134"]["inputs"].update(blocks_to_swap=blocks, offload_img_emb=blocks > 0, offload_txt_emb=blocks > 0)
        p["122"]["inputs"]["load_device"] = ji.get("load_device", "offload_device" if blocks else "main_device")
        if ji.get("base_precision"):
            p["122"]["inputs"]["base_precision"] = ji["base_precision"]
        p["128"]["inputs"]["force_offload"] = ji.get("force_offload", True)
        gv(Ws(messages(world["cached"])), p)
        return world["result"] or {"video": "b64"}

    gh, gv = job_guard.install(inner, get_videos)
    world["run"] = lambda job_input: gh({"id": "j", "input": job_input})
    return world


def job(**extra):
    return {"prompt": "hi", "image_url": "https://x/i.png", "wav_url": "https://x/a.wav", "width": 848,
            "height": 480, **extra}


def model_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[MODEL]")]


def test_five_jobs_load_once_then_reuse(worker, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    results = []
    for i, frames in enumerate([831, 3081, 831, 3081, 831]):     # 30 s and 2 min audio
        worker["frames"] = frames
        worker["cached"] = [] if i == 0 else ["122", "120", "134", "138", "177"]
        results.append(worker["run"](job()))
    assert [r["model"]["status"] for r in results] == ["loaded", "reused", "reused", "reused", "reused"]
    assert all(r["model"]["blocks_to_swap"] == 0 for r in results)
    assert [r["model"]["tiled_vae"] for r in results] == [False, True, False, True, False]
    sigs = {json.dumps(job_guard.loader_signature(p), sort_keys=True) for p in worker["queued"]}
    assert len(sigs) == 1                                          # identical loader chain on every job
    lines = model_lines(caplog)
    assert sum("MODEL LOADED (cold start)" in m for m in lines if "job" not in m) == 1
    assert sum(m.startswith("[MODEL] MODEL REUSED") for m in lines) == 4
    assert not [m for m in lines if "reloaded" in m]
    assert model_lock.STATE.loads == 1 and model_lock.STATE.reuses == 4


def test_model_setting_fields_are_ignored_and_logged(worker, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    worker["run"](job())
    worker["cached"] = ["122"]
    r = worker["run"](job(blocks_to_swap=25, force_offload=True, torch_compile=False, benchmark=True,
                          fp8_fast=True, attention_mode="sdpa", base_precision="fp16_fast"))
    assert r["model"]["status"] == "reused"
    p = worker["queued"][-1]
    assert p["134"]["inputs"]["blocks_to_swap"] == 0 and p["122"]["inputs"]["quantization"] == "fp8_e4m3fn"
    assert p["122"]["inputs"]["attention_mode"] == "sageattn" and p["122"]["inputs"]["base_precision"] == "bf16"
    assert p["122"]["inputs"].get("compile_args") == ["177", 0]
    assert any("ignored job fields" in m and "blocks_to_swap=25" in m for m in caplog.messages)


def test_unexpected_reload_is_reported(worker, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    worker["run"](job())
    worker["cached"] = []                                         # ComfyUI ran the loader again
    r = worker["run"](job())
    assert r["model"]["status"] == "reloaded"
    assert any("expected MODEL REUSED but ComfyUI ran the model loader again" in m for m in caplog.messages)


def test_signature_mismatch_refuses_the_job(worker):
    worker["run"](job())
    model_lock.STATE.signature = {"122": ["WanVideoModelLoader", {"base_precision": "fp16"}]}
    before = len(worker["queued"])
    r = worker["run"](job())
    assert "refusing to load a second copy" in r["error"] and r.get("refresh_worker") is True
    assert len(worker["queued"]) == before                        # nothing queued


def test_oom_is_not_retried_with_other_settings(worker):
    calls = []
    worker["result"] = {"error": "ComfyUI execution_error", "details": {
        "exception_type": "torch.OutOfMemoryError", "exception_message": "CUDA out of memory", "node_id": "128",
        "node_type": "WanVideoSampler"}}
    orig = worker["run"]
    r = orig(job())
    calls.append(len(worker["queued"]))
    assert calls == [1]                                           # one attempt, no blocks_to_swap ladder
    assert r.get("refresh_worker") is True and "not retried with other settings" in r["error"]
    assert model_lock.STATE.signature is None                     # ComfyUI dropped the model with the OOM


def test_not_enough_vram_fails_fast_before_queueing(worker):
    worker["run"](job())
    worker["used_gb"] = 30.5                                      # something else holds the VRAM
    worker["frames"] = 3081
    worker["cached"] = ["122"]
    before = len(worker["queued"])
    r = worker["run"](job())
    assert "not enough VRAM" in r["error"] and r.get("refresh_worker") is True
    assert worker["clears"] >= 1 and len(worker["queued"]) == before


def test_multi_and_720p_are_rejected_before_any_work(worker):
    assert "multi-person" in worker["run"](job(person_count="multi"))["error"]
    assert "larger than the locked" in worker["run"](job(width=1280, height=720))["error"]
    assert worker["queued"] == []


def test_model_lock_off_keeps_the_old_path(worker, monkeypatch, caplog):
    monkeypatch.setenv("MODEL_LOCK", "0")
    caplog.set_level(logging.INFO, logger="worker_guard")
    r = worker["run"](job())
    assert "model" not in r and not model_lines(caplog)
