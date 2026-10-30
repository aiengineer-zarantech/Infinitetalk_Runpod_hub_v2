# ==============================================================================
# Regression: the existing upload-audio video flow through the REAL handler.handler is
# unchanged by worker_guard. Runs the same job guarded and unguarded (functools __wrapped__)
# and compares the queued workflow and the returned {"video": base64}. ComfyUI is mocked.
# ==============================================================================
import base64
import copy
import json
import re
import urllib.request
from pathlib import Path

import pytest

import handler

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 fake mp4 payload"


class FakeWs:
    def __init__(self):
        self.messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}}),
            json.dumps({"type": "executing", "data": {"node": "128", "prompt_id": "p1"}}),
            *[json.dumps({"type": "progress", "data": {"value": v, "max": 12, "node": "128", "prompt_id": "p1"}})
              for v in range(1, 13)],  # 2 windows x 6 steps, like the InfiniteTalk loop
            b"\x01preview-frame",
            json.dumps({"type": "executing", "data": {"node": "131", "prompt_id": "p1"}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
        ]
        self.closed = False

    def connect(self, url):
        self.url = url

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture
def mocked_comfy(tmp_path, monkeypatch):
    video = tmp_path / "out" / "InfiniteTalk_00001.mp4"
    video.parent.mkdir()
    video.write_bytes(VIDEO_BYTES)
    queued = []

    def fake_queue(prompt, input_type="image", person_count="single"):
        queued.append(copy.deepcopy(prompt))
        return {"prompt_id": "p1"}

    monkeypatch.setattr(handler, "COMFY_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setattr(handler, "get_workflow_path", lambda it, pc: str(REPO_ROOT / ("I2V_single.json" if it == "image" else "V2V_single.json")))
    # handler.load_workflow uses open() without encoding: UTF-8 in the Linux image, cp1252 on Windows
    monkeypatch.setattr(handler, "load_workflow", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(handler, "get_audio_duration", lambda path: 4.0)
    monkeypatch.setattr(handler, "queue_prompt", fake_queue)
    monkeypatch.setattr(handler, "get_history", lambda pid: {pid: {"outputs": {"131": {"gifs": [{"fullpath": str(video)}]}}}})
    monkeypatch.setattr(handler.websocket, "WebSocket", FakeWs)
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: object())  # handler's HTTP readiness probe
    return queued


def job(**extra):
    payload = {
        "image_base64": base64.b64encode(b"\x89PNG fake image").decode(),
        "wav_base64": base64.b64encode(b"RIFF fake wav").decode(),
        "prompt": "A person talking naturally",
        "width": 512,
        "height": 768,
    }
    payload.update(extra)
    return {"input": payload}


def run_unguarded(monkeypatch, job_payload):
    monkeypatch.setattr(handler, "get_videos", handler.get_videos.__wrapped__)
    return handler.handler.__wrapped__(job_payload)


def strip_task_ids(prompt):
    """Input filenames contain a random task_<uuid>; normalise them for comparison."""
    text = json.dumps(prompt, sort_keys=True)
    import re
    return re.sub(r"task_[0-9a-f-]{36}", "task_X", text)


def test_handler_is_guarded():
    assert hasattr(handler.handler, "__wrapped__")
    assert hasattr(handler.get_videos, "__wrapped__")


APP_OFFLOAD = {"blocks_to_swap": 0, "force_offload": False}  # what the studio backend sends today


@pytest.mark.parametrize("extra", [
    {"offload_mode": "manual", **APP_OFFLOAD},
    {"offload_mode": "manual", "max_frame": 121, **APP_OFFLOAD},
    # 720p: 480p on >= 28 GB never offloads, even in manual mode
    {"offload_mode": "manual", "blocks_to_swap": 30, "force_offload": True, "load_device": "main_device",
     "width": 1280, "height": 720},
])
def test_manual_mode_identical_with_and_without_guard(mocked_comfy, monkeypatch, extra):
    monkeypatch.setenv("TORCH_COMPILE", "0")  # compile off: the original workflow exactly
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")  # manual mode is ignored unless explicitly allowed
    guarded = handler.handler(job(**extra))
    unguarded = run_unguarded(monkeypatch, job(**extra))

    expected = {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert guarded == expected
    assert unguarded == expected
    assert len(mocked_comfy) == 2
    assert strip_task_ids(mocked_comfy[0]) == strip_task_ids(mocked_comfy[1])

    prompt = mocked_comfy[0]
    assert prompt["270"]["inputs"]["value"] == extra.get("max_frame", int(4.0 * 25) + 81)
    assert prompt["245"]["inputs"]["value"] == extra.get("width", 512)
    assert prompt["246"]["inputs"]["value"] == extra.get("height", 768)
    assert prompt["125"]["inputs"]["audio"].endswith("_input_audio.wav")
    assert prompt["284"]["inputs"]["image"].endswith("_input_image.png")
    assert prompt["241"]["inputs"]["positive_prompt"] == "A person talking naturally"
    assert prompt["134"]["inputs"]["blocks_to_swap"] == extra["blocks_to_swap"]
    assert prompt["128"]["inputs"]["force_offload"] == extra["force_offload"]
    if "load_device" in extra:
        assert prompt["122"]["inputs"]["load_device"] == extra["load_device"]


@pytest.mark.parametrize("workflow", ["image", "video"])
def test_auto_mode_equals_unguarded_job_with_tier_values(mocked_comfy, monkeypatch, workflow):
    monkeypatch.setenv("TORCH_COMPILE", "0")
    # Offloading off by default: blocks_to_swap 0, force_offload False at any resolution.
    # The client's values (and a stray load_device) are ignored.
    media = {"video_base64": base64.b64encode(b"fake mp4").decode(), "input_type": "video"} if workflow == "video" else {}
    guarded = handler.handler(job(blocks_to_swap=30, force_offload=True, load_device="offload_device", **media))
    unguarded = run_unguarded(monkeypatch, job(**APP_OFFLOAD, **media))
    assert guarded == unguarded == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert strip_task_ids(mocked_comfy[0]) == strip_task_ids(mocked_comfy[1])
    prompt = mocked_comfy[0]
    assert prompt["134"]["inputs"]["blocks_to_swap"] == 0
    assert prompt["134"]["inputs"]["offload_img_emb"] is False
    assert prompt["128"]["inputs"]["force_offload"] is False
    assert prompt["122"]["inputs"]["load_device"] == "main_device"


def test_video_flow_logs_job_line(mocked_comfy, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="worker_guard")
    handler.handler(job())
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[JOB]")]
    assert len(lines) == 1
    assert lines[0].startswith("[JOB] done in ")
    assert re.search(
        r"\(frames 181, resolution 512x768, gpu NVIDIA GeForce RTX 5090, vram 32\.0 GB, peak 1\.0 GB, "
        r"blocks_to_swap 0, force_offload False, mode auto, oom_retry no, torch_compile on, compile ~\d+\.\ds, "
        r"fp8_fast off, avg \d+\.\d\d s/it, sampling \d+\.\ds\)$", lines[0]), lines[0]
    speed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[SPEED]")]
    assert [m.split(":")[0] for m in speed] == ["[SPEED] window 1/2", "[SPEED] window 2/2"]
    offload_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[OFFLOAD]")]
    assert offload_lines == ["[OFFLOAD] auto: VRAM 32.0 GB, 512x768 -> blocks_to_swap 0, force_offload False"]


def test_network_volume_output_unchanged(mocked_comfy, monkeypatch, tmp_path):
    copies = []
    monkeypatch.setattr(handler.shutil, "copy2", lambda src, dst: copies.append(dst))
    monkeypatch.setattr(handler.os.path, "getsize", lambda p: 10)
    res = handler.handler(job(network_volume=True))
    assert list(res) == ["video_path"] and res["video_path"].startswith("/runpod-volume/infinitetalk_task_")


# ---------------------------------------------------------------- torch.compile / fp8 / benchmark (v20)
def test_compile_on_equals_handler_loading_the_compiled_file(mocked_comfy, monkeypatch):
    guarded = handler.handler(job())  # TORCH_COMPILE defaults to 1
    monkeypatch.setattr(handler, "get_workflow_path", lambda it, pc: str(REPO_ROOT / "I2V_single_compiled.json"))
    unguarded = run_unguarded(monkeypatch, job(**APP_OFFLOAD))
    assert guarded == unguarded
    assert strip_task_ids(mocked_comfy[0]) == strip_task_ids(mocked_comfy[1])
    assert mocked_comfy[0]["122"]["inputs"]["compile_args"] == ["177", 0]


def test_compile_on_only_adds_compile_args(mocked_comfy, monkeypatch):
    handler.handler(job())
    monkeypatch.setenv("TORCH_COMPILE", "0")
    handler.handler(job())
    on, off = (json.loads(strip_task_ids(p)) for p in mocked_comfy)
    assert "compile_args" not in off["122"]["inputs"]
    on["122"]["inputs"].pop("compile_args")
    assert on == off


def test_v2v_has_no_compiled_variant(mocked_comfy, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="worker_guard")
    res = handler.handler(job(input_type="video", video_base64=base64.b64encode(b"fake mp4").decode()))
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert "compile_args" not in mocked_comfy[0]["122"]["inputs"]
    assert any("[COMPILE] off (no compiled variant of V2V_single.json)" in m for m in caplog.messages)
    assert "torch_compile off" in [m for m in caplog.messages if m.startswith("[JOB]")][0]


@pytest.mark.parametrize("fp8_env, expected", [(None, "fp8_e4m3fn"), ("0", "fp8_e4m3fn"), ("1", "fp8_e4m3fn_fast")])
def test_fp8_fast_only_when_enabled(mocked_comfy, monkeypatch, fp8_env, expected):
    if fp8_env is not None:
        monkeypatch.setenv("FP8_FAST", fp8_env)
    handler.handler(job())
    assert mocked_comfy[0]["122"]["inputs"]["quantization"] == expected


class CompileFailWs(FakeWs):
    """First websocket of the test reports a dynamo error at the sampler; later ones succeed."""
    failures = []

    def __init__(self):
        super().__init__()
        if CompileFailWs.failures:
            CompileFailWs.failures.pop()
            self.messages = [
                json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}}),
                json.dumps({"type": "execution_error", "data": {
                    "prompt_id": "p1", "node_id": "128", "node_type": "WanVideoSampler",
                    "exception_type": "torch._dynamo.exc.BackendCompilerFailed",
                    "exception_message": "backend='inductor' raised: no kernel image for sm_120", "traceback": []}}),
            ]


def test_compile_failure_reruns_once_with_original_workflow(mocked_comfy, monkeypatch, caplog):
    import logging
    from worker_guard import accel, comfy_api
    caplog.set_level(logging.INFO, logger="worker_guard")
    freed = []
    monkeypatch.setattr(comfy_api, "free", lambda **k: freed.append(1))
    monkeypatch.setattr(handler.websocket, "WebSocket", CompileFailWs)
    CompileFailWs.failures = [1]
    res = handler.handler(job())
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert [("compile_args" in p["122"]["inputs"]) for p in mocked_comfy] == [True, False]
    assert freed == [1]
    assert any(m.startswith("[COMPILE] failed: torch._dynamo.exc.BackendCompilerFailed") for m in caplog.messages)
    assert "compile_fallback yes" in [m for m in caplog.messages if m.startswith("[JOB]")][-1]
    # compile stays off on this worker afterwards
    handler.handler(job())
    assert "compile_args" not in mocked_comfy[2]["122"]["inputs"]
    assert accel._compile_disabled_reason


def test_low_compute_capability_turns_compile_off(mocked_comfy, monkeypatch, tmp_path):
    info = tmp_path / "gpu.json"
    info.write_text(json.dumps({"name": "NVIDIA A100-SXM4-80GB", "capability": "8.0"}))
    monkeypatch.setenv("WORKER_GPU_INFO", str(info))
    handler.handler(job())
    assert "compile_args" not in mocked_comfy[0]["122"]["inputs"]


def _wav_b64(seconds, rate=16000):
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


def test_benchmark_trims_audio_to_10s(mocked_comfy, monkeypatch):
    import wave
    durations = []

    def real_duration(path):
        with wave.open(path, "rb") as w:
            durations.append(w.getnframes() / w.getframerate())
        return durations[-1]

    monkeypatch.setattr(handler, "get_audio_duration", real_duration)
    handler.handler(job(wav_base64=_wav_b64(30), benchmark=True))
    assert durations == [10.0]
    assert mocked_comfy[0]["270"]["inputs"]["value"] == 10 * 25 + 81


def test_benchmark_job_can_switch_compile_per_job(mocked_comfy):
    handler.handler(job(wav_base64=_wav_b64(3), benchmark=True, torch_compile=False))
    handler.handler(job(wav_base64=_wav_b64(3), benchmark=True, torch_compile=True))
    handler.handler(job(wav_base64=_wav_b64(3), torch_compile=False))  # ignored without benchmark
    assert [("compile_args" in p["122"]["inputs"]) for p in mocked_comfy] == [False, True, True]
