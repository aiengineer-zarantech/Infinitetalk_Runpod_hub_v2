# ==============================================================================
# worker_guard/job_guard.py: per-job GPU guard, lazy Multi model, progress logs,
# VRAM watchdog, job timeout and [JOB] timing line. ComfyUI is fully mocked.
# ==============================================================================
import json
import logging
import threading
import types
import urllib.error

import pytest

import setup_models
from worker_guard import comfy_api, job_guard

MB = 1024 ** 2
GB = 1024 ** 3
MULTI = job_guard.MULTI_MODEL

PROMPT = {
    "128": {"class_type": "WanVideoSampler", "inputs": {}},
    "131": {"class_type": "VHS_VideoCombine", "inputs": {}},
    "245": {"class_type": "INTConstant", "inputs": {"value": 512}},
    "246": {"class_type": "INTConstant", "inputs": {"value": 768}},
    "270": {"class_type": "INTConstant", "inputs": {"value": 181}},
}


def stats(device_type="cuda", used_mb=20000, total_gb=32, name="cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync"):
    total = total_gb * GB
    return {"devices": [{"type": device_type, "name": name, "index": 0,
                         "vram_total": total, "vram_free": total - used_mb * MB}]}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(job_guard, "_now", c.now)
    monkeypatch.setattr(job_guard, "_sleep", c.sleep)
    return c


@pytest.fixture
def comfy(monkeypatch):
    """Mutable fake ComfyUI: stats, interrupts, object_info and /models listings."""
    state = types.SimpleNamespace(
        stats=stats(), stats_error=None, interrupts=[], object_info=None, object_info_error=None,
        models=[], calls=[],
    )
    state.object_info = {job_guard.MULTI_LOADER_CLASS: {"input": {"required": {"model": [[MULTI], {}]}}}}

    def system_stats(timeout=10):
        state.calls.append("system_stats")
        if state.stats_error:
            raise state.stats_error
        return state.stats

    def object_info(node_class, timeout=30):
        state.calls.append(f"object_info:{node_class}")
        if state.object_info_error:
            raise state.object_info_error
        return state.object_info

    def list_models(folder, timeout=30):
        state.calls.append(f"models:{folder}")
        return state.models

    monkeypatch.setattr(comfy_api, "system_stats", system_stats)
    monkeypatch.setattr(comfy_api, "interrupt", lambda prompt_id=None, timeout=10: state.interrupts.append(prompt_id))
    monkeypatch.setattr(comfy_api, "object_info", object_info)
    monkeypatch.setattr(comfy_api, "list_models", list_models)
    return state


@pytest.fixture
def ensure_calls(monkeypatch):
    calls = []

    def fake_ensure(name, subfolder=None):
        calls.append(name)
        return {"model": name, "copied": True, "size_mb": 2600.0, "copy_seconds": 20.0, "speed_mb_s": 130.0,
                "seconds": 20.5, "dest_path": f"/ComfyUI/models/diffusion_models/{name}",
                "local_path": f"/models_local/diffusion_models/{name}"}

    monkeypatch.setattr(setup_models, "ensure_model", fake_ensure)
    return calls


def install(inner=None, get_videos=None):
    inner = inner or (lambda job: {"video": "b64"})
    get_videos = get_videos or (lambda ws, prompt, input_type="image", person_count="single": {"131": ["/out.mp4"]})
    return job_guard.install(inner, get_videos)


def make_ctx(**overrides):
    settings = job_guard.Settings.from_env()
    if overrides:
        settings = job_guard.Settings(**{**settings.__dict__, **overrides})
    ctx = job_guard.JobContext(settings, "video")
    ctx.record_prompt(PROMPT)
    return ctx


# ---------------------------------------------------------------- install / switch
def test_disabled_returns_functions_unchanged(monkeypatch):
    monkeypatch.setenv("WORKER_GUARD_ENABLED", "0")
    inner, gv = object(), object()
    assert job_guard.install(inner, gv) == (inner, gv)


def test_wrapped_functions_keep_originals():
    def handler(job):
        return {}

    def get_videos(ws, prompt, input_type="image", person_count="single"):
        return {}

    gh, gv = job_guard.install(handler, get_videos)
    assert gh.__wrapped__ is handler and gv.__wrapped__ is get_videos


# ---------------------------------------------------------------- step 6: job GPU guard
@pytest.mark.parametrize("job_input", [
    {"image_base64": "x"},
    {"task": "tts", "text": "hello"},
])
def test_non_cuda_device_returns_refresh_and_skips_job(comfy, clock, job_input):
    comfy.stats = stats(device_type="cpu", name="cpu")
    called = []
    gh, _ = install(inner=lambda job: called.append(job) or {"video": "x"})
    res = gh({"input": job_input})
    assert res["refresh_worker"] is True
    assert res["error"].startswith("[GPU] ComfyUI device is 'cpu'")
    assert "Workflow not queued" in res["error"]
    assert called == []


def test_no_devices_reported_returns_refresh(comfy, clock):
    comfy.stats = {"devices": []}
    gh, _ = install()
    res = gh({"input": {}})
    assert res["refresh_worker"] is True and "'none'" in res["error"]


def test_unreachable_comfy_retries_then_refreshes(comfy, clock):
    comfy.stats_error = urllib.error.URLError("connection refused")
    called = []
    gh, _ = install(inner=lambda job: called.append(job))
    start = clock.t
    res = gh({"input": {"image_base64": "x"}})
    assert res["refresh_worker"] is True
    assert "unreachable for 60s" in res["error"]
    assert clock.t - start >= 60
    assert called == []


def test_comfy_recovers_within_retry_window(comfy, clock, monkeypatch):
    attempts = {"n": 0}

    def flaky(timeout=10):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("starting")
        return stats()

    monkeypatch.setattr(comfy_api, "system_stats", flaky)
    gh, _ = install()
    assert gh({"input": {}}) == {"video": "b64"}


def test_cmd_diagnostics_skip_guard(comfy, clock):
    comfy.stats = stats(device_type="cpu")
    gh, _ = install(inner=lambda job: {"cmd": job["input"]["cmd"], "output": "ok"})
    assert gh({"input": {"cmd": "nvidia-smi"}}) == {"cmd": "nvidia-smi", "output": "ok"}
    assert comfy.calls == []


def test_healthy_job_result_unchanged(comfy, clock):
    gh, _ = install()
    assert gh({"input": {"image_base64": "x"}}) == {"video": "b64"}


def test_inner_exception_is_reraised(comfy, clock):
    def boom(job):
        raise ValueError("bad input")

    gh, _ = install(inner=boom)
    with pytest.raises(ValueError, match="bad input"):
        gh({"input": {}})


# ---------------------------------------------------------------- lazy Multi model
def test_multi_job_prepares_model_before_inner(comfy, clock, ensure_calls):
    order = []
    gh, _ = install(inner=lambda job: order.append("inner") or {"video": "x"})
    comfy_calls_before = len(comfy.calls)
    res = gh({"input": {"person_count": "multi"}})
    assert res == {"video": "x"}
    assert ensure_calls == [MULTI]
    assert f"object_info:{job_guard.MULTI_LOADER_CLASS}" in comfy.calls[comfy_calls_before:]
    assert order == ["inner"]


def test_single_and_tts_jobs_do_not_copy_multi(comfy, clock, ensure_calls):
    gh, _ = install()
    gh({"input": {"person_count": "single"}})
    gh({"input": {"task": "tts", "person_count": "multi"}})
    assert ensure_calls == []


def test_multi_not_listed_by_comfy_blocks_queue(comfy, clock, ensure_calls):
    comfy.object_info = {job_guard.MULTI_LOADER_CLASS: {"input": {"required": {"model": [["other.safetensors"], {}]}}}}
    called = []
    gh, _ = install(inner=lambda job: called.append(1))
    res = gh({"input": {"person_count": "multi"}})
    assert res["error"].startswith(f"[MODELS] ComfyUI does not list {MULTI}")
    assert "refresh_worker" not in res
    assert called == []


def test_multi_new_combo_format_is_understood(comfy, clock, ensure_calls):
    comfy.object_info = {job_guard.MULTI_LOADER_CLASS: {"input": {"required": {"model": ["COMBO", {"options": [MULTI]}]}}}}
    gh, _ = install()
    assert gh({"input": {"person_count": "multi"}}) == {"video": "b64"}


def test_multi_falls_back_to_models_endpoint(comfy, clock, ensure_calls):
    comfy.object_info_error = urllib.error.HTTPError("u", 404, "not found", None, None)
    comfy.models = [MULTI]
    gh, _ = install()
    assert gh({"input": {"person_count": "multi"}}) == {"video": "b64"}
    assert f"models:{job_guard.MODEL_FOLDER}" in comfy.calls


def test_multi_copy_failure_blocks_queue(comfy, clock, monkeypatch):
    def failing(name, subfolder=None):
        raise setup_models.DiskSpaceError("disk full")

    monkeypatch.setattr(setup_models, "ensure_model", failing)
    called = []
    gh, _ = install(inner=lambda job: called.append(1))
    res = gh({"input": {"person_count": "multi"}})
    assert "failed to prepare" in res["error"] and "disk full" in res["error"]
    assert called == []


# ---------------------------------------------------------------- step 7: progress logs
def progress(value, total=6, node="128", prompt_id="p1"):
    return json.dumps({"type": "progress", "data": {"value": value, "max": total, "node": node, "prompt_id": prompt_id}})


class ScriptedWs:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    def recv(self):
        return self.messages.pop(0)

    def shutdown(self):
        self.closed = True


def test_progress_logs_are_throttled(clock, caplog):
    ctx = make_ctx()
    ws = ScriptedWs([progress(i) for i in range(1, 7)] + [b"\x00binary preview"])
    proxy = job_guard.WsProxy(ws, ctx)
    caplog.set_level(logging.INFO, logger="worker_guard")
    received = []
    for step in (0, 1, 2, 6, 7, 12):  # seconds since start
        clock.t = ctx.started + step
        received.append(proxy.recv())
    received.append(proxy.recv())
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[PROGRESS]")]
    assert len(lines) == 3  # t=0, t=6, t=12
    assert lines[0] == "[PROGRESS] node 128 (WanVideoSampler) 1/6 (elapsed 0s)"
    assert received[-1] == b"\x00binary preview"  # passed through unchanged
    assert received[0] == progress(1)


def test_proxy_tracks_execution_state():
    ctx = make_ctx()
    ws = ScriptedWs([
        json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}}),
        json.dumps({"type": "executing", "data": {"node": "131", "prompt_id": "p1"}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
    ])
    proxy = job_guard.WsProxy(ws, ctx)
    proxy.recv()
    assert ctx.executing and ctx.prompt_id == "p1"
    proxy.recv()
    assert ctx.current_class() == "VHS_VideoCombine"
    proxy.recv()
    assert not ctx.executing and ctx.current_node is None


# ---------------------------------------------------------------- step 7: VRAM watchdog
def run_ticks(watchdog, clock, until, step=30):
    while clock.t < until:
        clock.t += step
        watchdog.tick()


def test_vram_near_zero_trips_interrupt_and_refresh(comfy, clock, caplog):
    ctx = make_ctx()
    ctx.executing, ctx.current_node, ctx.prompt_id = True, "128", "p1"
    comfy.stats = stats(used_mb=500)
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    caplog.set_level(logging.INFO, logger="worker_guard")
    run_ticks(wd, clock, ctx.started + 200)
    assert ctx.trip_kind is None  # exactly 180s low so far (first low reading at +30s): not "more than"
    run_ticks(wd, clock, ctx.started + 240)
    assert ctx.trip_kind == "vram_idle"
    assert ctx.refresh_worker is True
    assert comfy.interrupts == ["p1"]
    assert any(r.getMessage().startswith("[WATCHDOG] VRAM used 0.5 / 32.0 GB, node 128 (WanVideoSampler)")
               for r in caplog.records)

    res = job_guard._apply_trip(ctx, {"error": "ComfyUI execution_interrupted at node 128", "details": {}})
    assert res["error"].startswith("[WATCHDOG] VRAM near zero")
    assert "| ComfyUI: ComfyUI execution_interrupted" in res["error"]
    assert res["refresh_worker"] is True and res["details"] == {}


def test_vram_exempt_node_does_not_trip(comfy, clock):
    ctx = make_ctx()
    ctx.executing, ctx.current_node = True, "131"  # VHS_VideoCombine (ffmpeg on CPU)
    comfy.stats = stats(used_mb=300)
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    run_ticks(wd, clock, ctx.started + 600)
    assert ctx.trip_kind is None and comfy.interrupts == []


def test_vram_timer_resets_when_vram_recovers(comfy, clock):
    ctx = make_ctx()
    ctx.executing, ctx.current_node = True, "128"
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    for i in range(20):  # alternate low / high every 2 polls, 600s total
        comfy.stats = stats(used_mb=500 if (i // 2) % 2 == 0 else 18000)
        clock.t += 30
        wd.tick()
    assert ctx.trip_kind is None


def test_low_vram_before_execution_starts_does_not_trip(comfy, clock):
    ctx = make_ctx()
    comfy.stats = stats(used_mb=100)
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    run_ticks(wd, clock, ctx.started + 600)
    assert ctx.trip_kind is None


def test_device_change_during_job_trips(comfy, clock):
    ctx = make_ctx()
    ctx.executing = True
    comfy.stats = stats(device_type="cpu")
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    run_ticks(wd, clock, ctx.started + 30)
    assert ctx.trip_kind == "device" and ctx.refresh_worker is True


def test_stats_failure_during_job_is_logged_not_fatal(comfy, clock):
    ctx = make_ctx()
    ctx.executing = True
    comfy.stats_error = OSError("timed out")
    wd = job_guard.Watchdog(ctx, ScriptedWs([]))
    run_ticks(wd, clock, ctx.started + 600)
    assert ctx.trip_kind is None


# ---------------------------------------------------------------- step 7: job timeout
def test_timeout_interrupts_then_refreshes_if_stuck(comfy, clock, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SEC", "100")
    ctx = make_ctx()
    ws = ScriptedWs([])
    wd = job_guard.Watchdog(ctx, ws)
    clock.t = ctx.started + 99
    wd.tick()
    assert ctx.trip_kind is None
    clock.t = ctx.started + 100
    wd.tick()
    assert ctx.trip_kind == "timeout" and ctx.refresh_worker is False
    assert len(comfy.interrupts) == 1
    clock.t += 59
    wd.tick()
    assert not ws.closed
    clock.t += 1
    wd.tick()
    assert ws.closed and ctx.refresh_worker is True
    res = job_guard._apply_trip(ctx, {"error": "WebSocketConnectionClosedException: closed"})
    assert res["error"].startswith("[WATCHDOG] job exceeded JOB_TIMEOUT_SEC=100s")
    assert res["refresh_worker"] is True


def test_timeout_before_queue_raises(comfy, clock, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SEC", "10")
    queued = []

    def inner(job):
        clock.t += 11  # slow input download
        try:
            return gv(None, PROMPT)
        except TimeoutError as e:  # the handler's own except-Exception path
            return {"error": f"TimeoutError: {e}"}

    gh, gv = install(inner=inner, get_videos=lambda *a, **k: queued.append(1))
    res = gh({"input": {}})
    assert queued == []
    assert res["error"].startswith("[WATCHDOG] job exceeded JOB_TIMEOUT_SEC=10s before the workflow was queued")
    assert "refresh_worker" not in res


def test_finished_video_is_kept_if_trip_races_completion(clock):
    ctx = make_ctx()
    ctx.trip("timeout", "[WATCHDOG] late", refresh=False)
    assert job_guard._apply_trip(ctx, {"video": "b64"}) == {"video": "b64"}


# ---------------------------------------------------------------- real threads (short timings)
class BlockingWs:
    """recv() blocks until ComfyUI 'reacts' to /interrupt, or the socket is shut down."""

    def __init__(self):
        self.messages = [json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}})]
        self.wake = threading.Event()
        self.closed = False

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        self.wake.wait(10)
        if self.closed:
            raise ConnectionError("socket is already closed.")
        return json.dumps({"type": "execution_interrupted", "data": {"prompt_id": "p1", "node_id": "128"}})

    def shutdown(self):
        self.closed = True
        self.wake.set()


def loop_get_videos(ws, prompt, input_type="image", person_count="single"):
    while True:
        message = json.loads(ws.recv())
        if message["type"] == "execution_interrupted":
            raise RuntimeError("ComfyUI execution_interrupted")


def handler_like(gv_holder, ws):
    def inner(job):
        try:
            return gv_holder[0](ws, PROMPT)
        except Exception as e:  # mirrors handler.py's error dict
            return {"error": f"{type(e).__name__}: {e}"}
    return inner


@pytest.mark.parametrize("comfy_reacts, expect_refresh", [(True, False), (False, True)])
def test_watchdog_thread_end_to_end(comfy, monkeypatch, comfy_reacts, expect_refresh):
    monkeypatch.setenv("JOB_TIMEOUT_SEC", "1")
    monkeypatch.setenv("INTERRUPT_GRACE_SEC", "1")
    ws = BlockingWs()
    if comfy_reacts:
        monkeypatch.setattr(comfy_api, "interrupt", lambda prompt_id=None, timeout=10: ws.wake.set())
    holder = []
    gh, gv = job_guard.install(handler_like(holder, ws), loop_get_videos)
    holder.append(gv)
    res = gh({"input": {}})
    assert res["error"].startswith("[WATCHDOG] job exceeded JOB_TIMEOUT_SEC=1s")
    assert bool(res.get("refresh_worker")) is expect_refresh
    assert ws.closed is expect_refresh


# ---------------------------------------------------------------- step 8: [JOB] line
def job_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[JOB]")]


def test_job_line_for_video(comfy, clock, caplog, monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE", "0")
    caplog.set_level(logging.INFO, logger="worker_guard")

    def inner(job):
        clock.t += 42.0
        return gv(ScriptedWs([]), PROMPT)

    gh, gv = install(inner=inner, get_videos=lambda ws, prompt, *a: {"video": "b64"})
    gh({"input": {"width": 512, "height": 768}})
    assert job_lines(caplog) == [
        "[JOB] done in 42.0s (frames 181, resolution 512x768, gpu NVIDIA GeForce RTX 5090, vram 32.0 GB, "
        "peak 19.5 GB, blocks_to_swap 0, force_offload False, mode auto, oom_retry no, torch_compile off, "
        "fp8_fast off, avg n/a s/it, sampling n/a)"
    ]


def test_job_line_for_tts_and_failures(comfy, clock, caplog, monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE", "0")
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh, _ = install(inner=lambda job: {"audio_base64": "x"})
    gh({"input": {"task": "tts"}})
    gh_fail, _ = install(inner=lambda job: {"error": "audio missing"})
    gh_fail({"input": {"blocks_to_swap": 5, "force_offload": True}})  # client values ignored
    assert job_lines(caplog) == [
        "[JOB] done in 0.0s (task tts)",
        "[JOB] failed in 0.0s (frames n/a, resolution n/a, gpu NVIDIA GeForce RTX 5090, vram 32.0 GB, peak 19.5 GB, "
        "blocks_to_swap 0, force_offload False, mode auto, oom_retry no, torch_compile off, fp8_fast off, "
        "avg n/a s/it, sampling n/a): audio missing",
    ]
