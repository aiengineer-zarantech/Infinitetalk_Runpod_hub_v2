# ==============================================================================
# v26 job flow through the REAL handler.handler + worker_guard (ComfyUI mocked):
# - fast480: generation once, InfiniteTalk freed, upscale, SeedVR2 freed; the 720p video replaces the 480p one
# - any upscale failure (OOM after all retries included): the 480p video is returned with "upscale": "failed"
#   and the generation is NOT run again
# - "upscale": "none" and old modes: no upscale stage at all
# - "mode": "upscale_only": no generation, only the upscale
# - an OOM in a segment retries only that segment with smaller chunks/tiles
# ==============================================================================
import base64
import logging
import os
import re
import shutil

import pytest

import handler
from test_handler_video_regression import VIDEO_BYTES, job, mocked_comfy, run_unguarded  # noqa: F401
from worker_guard import comfy_api, upscale

UPSCALED = b"\x00\x00\x00\x18ftypmp42 upscaled 720p"
GB = 1024 ** 3


def fake_result(path, model="3b"):
    return upscale.Result(path, upscale.MODEL_FILES[model], (848, 480), (1280, 720), 181, "25/1", 12.3,
                          int(20 * GB), int(30 * GB), int(64 * GB), 337, 1)


@pytest.fixture
def frees(monkeypatch):
    calls = []
    monkeypatch.setattr(comfy_api, "free", lambda unload_models=True, free_memory=True, timeout=10:
                        calls.append((unload_models, free_memory)))
    return calls


@pytest.fixture
def upscale_ok(monkeypatch):
    calls = []

    def run(source, model, out_path, remaining, job_input=None):
        with open(source, "rb") as f:
            source_bytes = f.read()
        calls.append({"source": source, "source_bytes": source_bytes, "model": model, "out": out_path, "job": job_input})
        with open(out_path, "wb") as f:
            f.write(UPSCALED)
        return fake_result(out_path, model)

    monkeypatch.setattr(upscale, "run", run)
    return calls


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "comfy_output"
    out.mkdir()
    monkeypatch.setattr(upscale, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(upscale, "INPUT_DIR", str(tmp_path / "comfy_input"))
    return out


def test_fast480_is_upscaled_after_one_generation(mocked_comfy, upscale_ok, frees, output_dir, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    res = handler.handler(job(mode="fast480", upscale_model="7b"))
    assert res == {"video": base64.b64encode(UPSCALED).decode(), "upscale": "720p"}
    assert len(mocked_comfy) == 1                                   # InfiniteTalk ran once
    (call,) = upscale_ok
    assert call["model"] == "7b" and call["source"].endswith("InfiniteTalk_00001.mp4")
    assert call["out"].startswith(str(output_dir))
    assert len(frees) == 2                                          # before (InfiniteTalk) and after (SeedVR2)
    final = [m for m in caplog.messages if m.startswith("[JOB] fast480")]
    assert len(final) == 1 and re.fullmatch(
        r"\[JOB\] fast480 done: res 848x480 -> 1280x720, gen \d+\.\ds, upscale \d+\.\ds, total \d+\.\ds, "
        r"attention sageattn, blocks_to_swap 0, vram peak sampler n/a / vae n/a / upscale 20\.0 GB", final[0]), final
    assert "_return_video_file" not in mocked_comfy[0]
    assert not os.path.exists(call["out"])  # the 720p file is removed once returned


@pytest.mark.parametrize("error", [
    "segment 1/2: ComfyUI execution_error at node 9 (KSampler): OutOfMemoryError: CUDA out of memory",
    "output check failed: frames 180 != source 181",
])
def test_failed_upscale_returns_the_480p_video(mocked_comfy, frees, monkeypatch, caplog, error):
    caplog.set_level(logging.INFO, logger="worker_guard")

    def fail(*args, **kwargs):
        raise upscale.UpscaleError(error)

    monkeypatch.setattr(upscale, "run", fail)
    res = handler.handler(job(mode="fast480"))
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode(), "upscale": "failed", "upscale_error": error}
    assert len(mocked_comfy) == 1                                   # never re-generated for an upscale failure
    assert len(frees) == 2
    assert any(re.match(r"\[JOB\] fast480 done: res 848x480, gen \d+\.\ds, upscale \d+\.\ds \(failed\), ", m)
               for m in caplog.messages)


def test_unexpected_upscale_bug_still_returns_480p(mocked_comfy, frees, monkeypatch):
    monkeypatch.setattr(upscale, "run", lambda *a, **k: 1 / 0)
    res = handler.handler(job(mode="fast480"))
    assert res["video"] == base64.b64encode(VIDEO_BYTES).decode()
    assert res["upscale"] == "failed" and res["upscale_error"].startswith("ZeroDivisionError")


@pytest.mark.parametrize("extra", [{"mode": "fast480", "upscale": "none"}, {}, {"mode": "standard"},
                                   {"upscale": "none"}])
def test_no_upscale_stage(mocked_comfy, upscale_ok, frees, extra):
    res = handler.handler(job(**extra))
    assert res == {"video": base64.b64encode(VIDEO_BYTES).decode()}
    assert upscale_ok == [] and frees == []


@pytest.mark.parametrize("extra, message", [
    ({"upscale": "720p"}, "upscale is only available with mode 'fast480' or 'upscale_only'"),
    ({"mode": "standard", "upscale": "720p"}, "upscale is only available"),
    ({"mode": "fast480", "upscale": "1080p"}, "upscale must be '720p' or 'none'"),
    ({"mode": "fast480", "upscale_model": "13b"}, "upscale_model must be one of ['3b', '7b']"),
    ({"mode": "upscale_only"}, "upscale_only needs one of ['video_base64', 'video_url', 'video_path']"),
])
def test_invalid_upscale_inputs(mocked_comfy, extra, message):
    res = handler.handler(job(**extra))
    assert res["error"].startswith("[MODE] ") and message in res["error"]
    assert mocked_comfy == []


def test_upscale_only(mocked_comfy, upscale_ok, frees, monkeypatch, caplog, output_dir):
    caplog.set_level(logging.INFO, logger="worker_guard")
    monkeypatch.setattr(upscale, "probe", lambda path: upscale.Probe(181, "25/1", 848, 480, True))
    res = handler.handler({"input": {"mode": "upscale_only", "video_base64": base64.b64encode(VIDEO_BYTES).decode()}})
    assert res == {"video": base64.b64encode(UPSCALED).decode()}
    assert mocked_comfy == []                                       # no InfiniteTalk prompt
    (call,) = upscale_ok
    assert call["model"] == "3b"
    assert call["source_bytes"] == VIDEO_BYTES
    assert not os.path.exists(call["source"]) and not os.path.exists(call["out"])  # temp files removed
    assert len(frees) == 2
    assert any(re.match(r"\[JOB\] upscale_only done: res 848x480 -> 1280x720, gen n/a, upscale \d+\.\ds, ", m)
               for m in caplog.messages), [m for m in caplog.messages if m.startswith("[JOB]")]


def test_upscale_only_failure_is_an_error(mocked_comfy, frees, monkeypatch):
    monkeypatch.setattr(upscale, "probe", lambda path: upscale.Probe(181, "25/1", 848, 480, True))

    def fail(*args, **kwargs):
        raise upscale.UpscaleError("model files not ready: FileNotFoundError")

    monkeypatch.setattr(upscale, "run", fail)
    res = handler.handler({"input": {"mode": "upscale_only", "video_base64": base64.b64encode(b"x").decode()}})
    assert res == {"error": "[UPSCALE] failed: model files not ready: FileNotFoundError"}


def test_upscale_only_needs_the_guard(mocked_comfy, monkeypatch):
    res = run_unguarded(monkeypatch, {"input": {"mode": "upscale_only", "video_base64": "eA=="}})
    assert res == {"error": "[MODE] upscale_only needs the worker guard (WORKER_GUARD_ENABLED=1)"}


# ---------------------------------------------------------------- OOM retries inside the upscale stage

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (present in the worker image)")


def _oom(node="9"):
    return upscale.SegmentError({"type": "execution_error", "node_id": node, "node_type": "KSampler",
                                 "exception_type": "torch.OutOfMemoryError",
                                 "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"})


@needs_ffmpeg
def test_oom_retries_only_the_segment(tmp_path, monkeypatch, frees, caplog):
    from test_upscale_frames import FakeComfy, make_clip
    caplog.set_level(logging.INFO, logger="worker_guard")
    src = make_clip(tmp_path / "src.mp4", (848, 480), 50)
    fake = FakeComfy(tmp_path)
    tries = []

    def flaky(prompt, remaining):
        tries.append(prompt["6"]["inputs"].get("chunking_mode.frames_per_chunk"))
        if len(tries) in (2, 3):  # segment 2 runs out of memory twice
            raise _oom()
        return fake(prompt, remaining)

    monkeypatch.setattr(upscale, "ensure_models", lambda model_key, remaining: None)
    monkeypatch.setattr(upscale, "run_prompt", flaky)
    result = upscale.run(src, "3b", str(tmp_path / "out.mp4"), lambda: 3600, {"upscale_segment_frames": 21})
    assert result.frames == 50
    assert tries == [None, None, 21, 13, None]       # auto, OOM, chunk 21 OOM, chunk 13 OK, next segment auto
    assert frees == [(False, True), (False, True)]   # caches freed between retries, models kept
    assert any(m.startswith("[UPSCALE] OOM in segment 2/3: retry 1/3 with chunk 21 frames, tile 512") for m in caplog.messages)
    assert any(m.startswith("[UPSCALE] segment 2/3: frames 21-41 ") and m.endswith("after 2 OOM retries")
               for m in caplog.messages)


@needs_ffmpeg
def test_oom_after_all_retries_fails_the_upscale(tmp_path, monkeypatch, frees):
    from test_upscale_frames import make_clip
    src = make_clip(tmp_path / "src.mp4", (848, 480), 20)
    tries = []

    def always_oom(prompt, remaining):
        tries.append(1)
        raise _oom()

    monkeypatch.setattr(upscale, "ensure_models", lambda model_key, remaining: None)
    monkeypatch.setattr(upscale, "run_prompt", always_oom)
    with pytest.raises(upscale.UpscaleError, match=r"segment 1/1: .*OutOfMemoryError"):
        upscale.run(src, "3b", str(tmp_path / "out.mp4"), lambda: 3600, {})
    assert len(tries) == 1 + len(upscale.OOM_RETRIES)


@needs_ffmpeg
def test_other_errors_are_not_retried(tmp_path, monkeypatch, frees):
    from test_upscale_frames import make_clip
    src = make_clip(tmp_path / "src.mp4", (848, 480), 20)
    tries = []

    def broken(prompt, remaining):
        tries.append(1)
        raise upscale.SegmentError({"node_id": "7", "node_type": "UNETLoader", "exception_type": "RuntimeError",
                                    "exception_message": "bad file"})

    monkeypatch.setattr(upscale, "ensure_models", lambda model_key, remaining: None)
    monkeypatch.setattr(upscale, "run_prompt", broken)
    with pytest.raises(upscale.UpscaleError, match="UNETLoader"):
        upscale.run(src, "3b", str(tmp_path / "out.mp4"), lambda: 3600, {})
    assert tries == [1]


# ---------------------------------------------------------------- run_prompt against a fake ComfyUI history

def test_run_prompt_success_error_and_rejection(tmp_path, monkeypatch):
    seg = tmp_path / "seg.mkv"
    seg.write_bytes(b"x")
    monkeypatch.setattr(upscale, "_sleep", lambda s: None)
    histories = iter([{}, {"p1": {"status": {"status_str": "success", "completed": True},
                                  "outputs": {"13": {"gifs": [{"fullpath": str(seg)}]}}}}])
    monkeypatch.setattr(comfy_api, "queue", lambda prompt, client_id, timeout=30: {"prompt_id": "p1", "node_errors": {}})
    monkeypatch.setattr(comfy_api, "history", lambda pid, timeout=10: next(histories))
    assert upscale.run_prompt({}, lambda: 100) == str(seg)

    error = {"p1": {"status": {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {}], ["execution_error", {"node_id": "9", "node_type": "KSampler",
                                                      "exception_type": "torch.OutOfMemoryError",
                                                      "exception_message": "CUDA out of memory"}]]}, "outputs": {}}}
    monkeypatch.setattr(comfy_api, "history", lambda pid, timeout=10: error)
    with pytest.raises(upscale.SegmentError) as e:
        upscale.run_prompt({}, lambda: 100)
    assert e.value.is_oom

    monkeypatch.setattr(comfy_api, "queue", lambda prompt, client_id, timeout=30:
                        {"prompt_id": "p2", "node_errors": {"6": {"errors": ["bad"]}}})
    with pytest.raises(upscale.UpscaleError, match="rejected the upscale workflow"):
        upscale.run_prompt({}, lambda: 100)


def test_run_prompt_stops_at_the_time_limit(monkeypatch):
    interrupted = []
    monkeypatch.setattr(comfy_api, "queue", lambda prompt, client_id, timeout=30: {"prompt_id": "p1"})
    monkeypatch.setattr(comfy_api, "interrupt", lambda prompt_id=None, timeout=10: interrupted.append(prompt_id))
    with pytest.raises(upscale.UpscaleError, match="time limit"):
        upscale.run_prompt({}, lambda: 0)
    assert interrupted == ["p1"]
