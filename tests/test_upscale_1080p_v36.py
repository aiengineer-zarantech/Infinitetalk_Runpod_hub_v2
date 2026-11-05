# ==============================================================================
# v36: SeedVR2 for the Studio's "Upscale to 1080p" (separate upscale endpoint, MODEL_LOCK=0)
# W1  setup_models.ensure_upscale_model: MODEL_SOURCE=hf_cache reads the HF snapshot FIRST, so the upscale
#     needs no network volume (and never downloads) when the files are in the cached repo
# W2  "upscale": "1080p" (upscale_only): exact target size, RAM per frame scaled to it, upscale_crf
# W3  job_guard._deliver: large results go to the S3 bucket (BUCKET_* env) as a presigned URL
# ==============================================================================
import base64
import logging
import os
import shutil
import sys
import types

import pytest

import setup_models
from test_setup_models_parallel import env  # noqa: F401
from test_setup_models_resilience import REPO, clean, use_cache  # noqa: F401
from test_setup_models_upscale import CONTENT, downloads, small_models  # noqa: F401
from worker_guard import job_guard, modes, upscale

HASH = "fedcba9876543210fedcba9876543210fedcba98"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg (present in the worker image)")


# ---------------------------------------------------------------- W1: HF cache first
def upscale_cache(tmp_path, names, data=None):
    """HF cache snapshot with the upscale files at <snapshot>/<ComfyUI folder>/<file> (the HF repo layout)."""
    root = tmp_path / "hf-hub"
    snap = root / ("models--" + REPO.replace("/", "--")) / "snapshots" / HASH
    for name in names:
        folder = setup_models.UPSCALE_MODELS[name]["folder"]
        (snap / folder).mkdir(parents=True, exist_ok=True)
        (snap / folder / name).write_bytes((data or CONTENT)[name])
    return root, snap


@pytest.fixture
def no_volume(env, monkeypatch, tmp_path):
    """The network volume is detached: nothing under SOURCE_ROOTS / VOLUME_ROOT."""
    shutil.rmtree(env.volume)
    monkeypatch.setattr(setup_models, "VOLUME_ROOT", str(tmp_path / "detached-volume"))
    return env


@pytest.fixture
def download_spy(monkeypatch):
    """Records every download_upscale_model call (the real function still runs, served by `downloads`)."""
    calls = []
    real = setup_models.download_upscale_model

    def spy(model_name, urlopen=None):
        calls.append(model_name)
        return real(model_name, urlopen)

    monkeypatch.setattr(setup_models, "download_upscale_model", spy)
    return calls


# (a) right size in the HF snapshot: linked from the cache, download_upscale_model never called, nothing copied
def test_hf_cache_is_used_without_a_volume_and_nothing_is_downloaded(no_volume, small_models, downloads,
                                                                     download_spy, monkeypatch, tmp_path, capsys):
    names = ["seedvr2_3b_fp16.safetensors", "seedvr2_ema_vae_fp16.safetensors"]
    root, snap = upscale_cache(tmp_path, names)
    use_cache(monkeypatch, root, HF_CACHE_REVISION=HASH, HF_CACHE_FALLBACK="0")
    for name in names:
        info = setup_models.ensure_upscale_model(name)
        folder = setup_models.UPSCALE_MODELS[name]["folder"]
        assert info["hf_cache"] is True and info["copied"] is False
        assert os.path.islink(info["dest_path"])
        assert os.path.realpath(info["dest_path"]) == os.path.realpath(snap / folder / name)
        assert info["dest_path"] == os.path.join(setup_models.COMFY_MODELS_DIR, folder, name)
        assert not os.path.exists(os.path.join(setup_models.LOCAL_MODELS_DIR, folder, name))   # no copy
    assert download_spy == [] and downloads["urls"] == []
    out = capsys.readouterr().out
    assert "[UPSCALE] seedvr2_3b_fp16.safetensors from HF cache, no download" in out
    assert "[UPSCALE] seedvr2_ema_vae_fp16.safetensors from HF cache, no download" in out
    assert "downloading" not in out


# (b) not in the snapshot: the old path (volume, else download once) runs
def test_hf_cache_missing_file_falls_back_to_the_download(env, small_models, downloads, download_spy,
                                                          monkeypatch, tmp_path, capsys):
    name = "seedvr2_3b_fp16.safetensors"
    root, _ = upscale_cache(tmp_path, ["seedvr2_ema_vae_fp16.safetensors"])
    use_cache(monkeypatch, root, HF_CACHE_REVISION=HASH, HF_CACHE_FALLBACK="0")
    info = setup_models.ensure_upscale_model(name)
    assert download_spy == [name] and len(downloads["urls"]) == 1
    assert info.get("hf_cache") is None and info["copied"] is True
    assert os.path.realpath(info["dest_path"]) == os.path.realpath(info["local_path"])
    assert "not in HF cache; falling back to volume/download" in capsys.readouterr().out


def test_hf_cache_missing_file_with_a_volume_copy_uses_the_volume(env, small_models, downloads, download_spy,
                                                                  monkeypatch, tmp_path):
    root, _ = upscale_cache(tmp_path, [])
    use_cache(monkeypatch, root, HF_CACHE_REVISION=HASH)
    path = env.volume / "models" / "upscale" / "seedvr2_ema_vae_fp16.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CONTENT["seedvr2_ema_vae_fp16.safetensors"])
    info = setup_models.ensure_upscale_model("seedvr2_ema_vae_fp16.safetensors")
    assert info.get("hf_cache") is None and info["copied"] is True
    assert download_spy == [] and downloads["urls"] == []


# (c) wrong size in the snapshot: fallback (download), and the rejected cached file is NOT the one linked
def test_hf_cache_file_with_the_wrong_size_falls_back(env, small_models, downloads, download_spy, monkeypatch,
                                                      tmp_path, capsys):
    name = "seedvr2_3b_fp16.safetensors"
    root, snap = upscale_cache(tmp_path, [name], data={name: b"short"})
    use_cache(monkeypatch, root, HF_CACHE_REVISION=HASH)
    info = setup_models.ensure_upscale_model(name)
    assert download_spy == [name]
    assert info.get("hf_cache") is None and info["copied"] is True
    dest = info["dest_path"]
    assert os.path.realpath(dest) != os.path.realpath(snap / "diffusion_models" / name)
    assert os.path.getsize(dest) == setup_models.UPSCALE_MODELS[name]["size"]
    assert "HF cache size 5 != " in capsys.readouterr().out


def test_hf_cache_unusable_falls_back(env, small_models, downloads, download_spy, monkeypatch, tmp_path, capsys):
    use_cache(monkeypatch, tmp_path / "no-cache-here", HF_CACHE_REVISION=HASH)
    info = setup_models.ensure_upscale_model("seedvr2_ema_vae_fp16.safetensors")
    assert download_spy == ["seedvr2_ema_vae_fp16.safetensors"] and info["copied"] is True
    assert "HF cache not usable" in capsys.readouterr().out


def test_upscale_job_path_goes_through_ensure_upscale_model(monkeypatch):
    """worker_guard/upscale.ensure_models is the only job path to the SeedVR2 files: it must call
    setup_models.ensure_upscale_model (never find_source / download_upscale_model directly)."""
    import inspect
    src = inspect.getsource(upscale.ensure_models)
    assert "setup_models.ensure_upscale_model(" in src
    assert "download_upscale_model" not in src and "find_source" not in src


def test_real_sizes_match_the_public_source():
    """The pinned Comfy-Org/SeedVR2 files (sizes / sha256 checked against the HF API on 2026-11-04)."""
    m = setup_models.UPSCALE_MODELS
    assert m["seedvr2_3b_fp16.safetensors"]["size"] == 6784268336
    assert m["seedvr2_ema_vae_fp16.safetensors"]["size"] == 501324814
    assert m["seedvr2_3b_fp16.safetensors"]["folder"] == "diffusion_models"
    assert m["seedvr2_ema_vae_fp16.safetensors"]["folder"] == "vae"
    assert m["seedvr2_ema_vae_fp16.safetensors"]["sha256"] == \
        "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"


# ---------------------------------------------------------------- W2: modes
UP = {"mode": "upscale_only", "video_url": "https://x/v.mp4"}


@pytest.mark.parametrize("extra", [
    {"upscale": "1080p"},
    {"upscale": "1080p", "upscale_width": 1920, "upscale_height": 1080, "upscale_crf": 14},
    {"upscale": "1080p", "upscale_width": 1440, "upscale_height": 1080, "output_upload": "bucket"},
    {"upscale": "1080P", "upscale_model": "7b"},
])
def test_1080p_inputs_are_accepted(extra):
    out, error = modes.prepare_job_input({**UP, **extra})
    assert error is None
    assert modes.upscale_settings(out) == {"model": str(extra.get("upscale_model", "3b"))}


@pytest.mark.parametrize("job_input, message", [
    ({"mode": "fast480", "upscale": "1080p"}, "upscale '1080p' is only available with mode 'upscale_only'"),
    ({**UP, "upscale": "4k"}, "upscale must be '720p', '1080p' or 'none'"),
    ({**UP, "upscale": "1080p", "upscale_width": 1920}, "must be sent together"),
    ({**UP, "upscale": "1080p", "upscale_width": 1921, "upscale_height": 1080}, "upscale_width must be an even"),
    ({**UP, "upscale": "1080p", "upscale_width": 1920, "upscale_height": 4000}, "upscale_height must be an even"),
    ({**UP, "upscale": "1080p", "upscale_width": "1920", "upscale_height": 1080}, "upscale_width must be an even"),
    ({**UP, "upscale_width": 1920, "upscale_height": 1080}, "need upscale '1080p'"),
    ({**UP, "upscale": "1080p", "upscale_crf": 60}, "upscale_crf must be an integer in 0..51"),
    ({**UP, "upscale": "1080p", "output_upload": "ftp"}, "output_upload must be one of"),
])
def test_invalid_1080p_inputs_are_rejected(job_input, message):
    _, error = modes.prepare_job_input(job_input)
    assert error and error.startswith("[MODE] ") and message in error, error


@pytest.mark.parametrize("job_input, size, expected", [
    ({"upscale": "720p"}, (848, 480), None),
    ({}, (848, 480), None),
    ({"upscale": "1080p"}, (848, 480), (1920, 1080)),
    ({"upscale": "1080p"}, (480, 848), (1080, 1920)),
    ({"upscale": "1080p", "upscale_width": 1080, "upscale_height": 1080}, (512, 512), (1080, 1080)),
])
def test_requested_size(job_input, size, expected):
    assert modes.requested_size(job_input, *size) == expected


# ---------------------------------------------------------------- W2: upscale.py
@pytest.mark.parametrize("size, requested", [
    ((848, 480), (1920, 1080)),     # 0.6 % off: center crop of a few pixels
    ((720, 406), (1920, 1080)),     # 3:2 already cropped to 16:9 by the Studio
    ((640, 480), (1440, 1080)),     # 4:3, the Studio pads it afterwards
    ((512, 512), (1080, 1080)),
])
def test_requested_target_is_used(size, requested):
    assert upscale.target_size(*size, requested) == requested


def test_requested_target_still_refuses_stretching():
    with pytest.raises(upscale.UpscaleError, match="crop or pad the input"):
        upscale.target_size(720, 480, (1920, 1080))           # 3:2 into 16:9 would stretch


def test_720p_rule_unchanged():
    assert upscale.target_size(848, 480) == (1280, 720)
    assert upscale.target_size(480, 848) == (720, 1280)


def test_ram_per_frame_scales_with_the_target(monkeypatch):
    monkeypatch.delenv("UPSCALE_RAM_PER_FRAME_MB", raising=False)
    assert upscale.ram_per_frame_mb() == 150.0
    assert upscale.ram_per_frame_mb((1280, 720)) == 150.0
    assert upscale.ram_per_frame_mb((1920, 1080)) == pytest.approx(337.5)
    assert upscale.ram_per_frame_mb((640, 360)) == 150.0      # never below the measurement
    monkeypatch.setenv("UPSCALE_RAM_PER_FRAME_MB", "200")
    assert upscale.ram_per_frame_mb((1920, 1080)) == 200.0


@pytest.mark.parametrize("target, size", [
    ((1280, 720), (320, 180)), ((1920, 1080), (320, 180)), ((1440, 1080), (320, 240)), ((1080, 1080), (320, 320)),
    ((1080, 1920), (180, 320)),
])
def test_frozen_check_size_for_any_target(target, size):
    assert upscale.check_size(target) == size


@needs_ffmpeg
@pytest.mark.parametrize("src_size, extra, target", [
    ((848, 480), {}, (1920, 1080)),
    ((640, 480), {"upscale_width": 1440, "upscale_height": 1080}, (1440, 1080)),
])
def test_run_1080p_end_to_end_with_a_fake_comfyui(tmp_path, monkeypatch, src_size, extra, target):
    """The real run(): prompts get the exact target, the result is verified at that size, crf is passed on."""
    from test_upscale_frames import FakeComfy, make_clip
    src = make_clip(tmp_path / "src.mp4", src_size, 30, audio=False)
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr(upscale, "ensure_models", lambda model_key, remaining: None)
    monkeypatch.setattr(upscale, "run_prompt", fake)
    crfs = []
    real_assemble = upscale.assemble

    def assemble(*args, **kwargs):
        crfs.append(args[5] if len(args) > 5 else kwargs.get("crf"))
        return real_assemble(*args, **kwargs)

    monkeypatch.setattr(upscale, "assemble", assemble)
    job = {"mode": "upscale_only", "upscale": "1080p", "upscale_crf": 14, **extra}
    result = upscale.run(src, "3b", str(tmp_path / "out.mp4"), lambda: 3600, job)
    assert result.out_size == target
    assert {(p["2"]["inputs"]["width"], p["2"]["inputs"]["height"]) for p in fake.prompts} == {target}
    out = upscale.probe(str(tmp_path / "out.mp4"))
    assert (out.width, out.height, out.frames, out.fps) == (*target, 30, "25/1")
    assert crfs == [14]


# ---------------------------------------------------------------- W3: delivery
BUCKET = {"BUCKET_ENDPOINT_URL": "https://s3.example", "BUCKET_ACCESS_KEY_ID": "id",
          "BUCKET_SECRET_ACCESS_KEY": "secret", "BUCKET_NAME": "studio-upscale"}


@pytest.fixture
def result_file(tmp_path):
    def make(size):
        path = tmp_path / f"result_{size}.mp4"
        path.write_bytes(b"v" * size)
        return str(path)
    return make


@pytest.fixture
def uploads(monkeypatch):
    calls = []

    def upload_file_to_bucket(file_name, file_location, bucket_creds=None, bucket_name=None, prefix=None,
                              extra_args=None):
        calls.append({"file_name": file_name, "bucket": bucket_name, "prefix": prefix,
                      "bytes": os.path.getsize(file_location)})
        return f"https://s3.example/{bucket_name}/{prefix}/{file_name}?X-Amz-Signature=abc"

    rp_upload = types.SimpleNamespace(upload_file_to_bucket=upload_file_to_bucket)
    utils = types.SimpleNamespace(rp_upload=rp_upload)
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", utils)
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils.rp_upload", rp_upload)
    for key in BUCKET:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("UPSCALE_BASE64_MAX_MB", "0.001")   # ~1 KB
    return calls


def test_small_result_stays_base64(result_file, uploads, monkeypatch):
    for k, v in BUCKET.items():
        monkeypatch.setenv(k, v)
    path = result_file(100)
    assert job_guard._deliver(path, False, "task_1", {}) == {"video": base64.b64encode(b"v" * 100).decode()}
    assert uploads == [] and not os.path.exists(path)


def test_large_result_goes_to_the_bucket(result_file, uploads, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    for k, v in BUCKET.items():
        monkeypatch.setenv(k, v)
    path = result_file(5000)
    out = job_guard._deliver(path, False, "task_2", {})
    assert out == {"video_url": "https://s3.example/studio-upscale/upscaled/task_2.mp4?X-Amz-Signature=abc",
                   "upload": "bucket"}
    assert uploads == [{"file_name": "task_2.mp4", "bucket": "studio-upscale", "prefix": "upscaled", "bytes": 5000}]
    assert not os.path.exists(path)
    assert not any("secret" in m for m in caplog.messages)


def test_large_result_without_bucket_is_base64_with_a_warning(result_file, uploads, caplog):
    path = result_file(5000)
    out = job_guard._deliver(path, False, "task_3", {})
    assert set(out) == {"video"} and uploads == []
    assert any("bucket not configured" in m for m in caplog.messages)


@pytest.mark.parametrize("mode, size, uploaded", [("bucket", 10, True), ("base64", 5000, False)])
def test_output_upload_overrides(result_file, uploads, monkeypatch, mode, size, uploaded):
    for k, v in BUCKET.items():
        monkeypatch.setenv(k, v)
    out = job_guard._deliver(result_file(size), False, "task_4", {"output_upload": mode})
    assert ("video_url" in out) is uploaded and ("video" in out) is not uploaded


def test_failed_upload_falls_back_to_base64(result_file, uploads, monkeypatch, caplog):
    for k, v in BUCKET.items():
        monkeypatch.setenv(k, v)

    def boom(**kwargs):
        raise RuntimeError("AccessDenied")

    sys.modules["runpod.serverless.utils"].rp_upload.upload_file_to_bucket = boom
    out = job_guard._deliver(result_file(5000), False, "task_5", {})
    assert set(out) == {"video"}
    assert any("bucket upload failed (RuntimeError: AccessDenied)" in m for m in caplog.messages)

