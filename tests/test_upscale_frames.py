# ==============================================================================
# v26: the upscaled video has exactly the source's frame count, frame rate and audio packets (-c:a copy),
# at exactly 1280x720 / 720x1280, with every frame in its original place. Real ffmpeg on synthetic clips;
# only ComfyUI is replaced by a fake that "upscales" each segment with ffmpeg (crop + scale, like ImageScale
# crop "center"). A segment that loses one frame must make the job's output check fail.
# ==============================================================================
import shutil
import subprocess

import numpy as np
import pytest

from worker_guard import upscale

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="needs ffmpeg/ffprobe (present in the worker image)")


def make_clip(path, size, frames, fps="25", audio=True):
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return str(path)


class FakeComfy:
    """Writes each segment like the SeedVR2 workflow would: frames [load_start, load_end), center-cropped to the
    target aspect, scaled to the target, lossless. drop=(segment index) loses a frame in the middle of it."""

    def __init__(self, tmp_path, drop=None):
        self.tmp_path = tmp_path
        self.drop = drop
        self.prompts = []

    def __call__(self, prompt, remaining):
        self.prompts.append(prompt)
        load = prompt["1"]["inputs"]
        src, skip, cap = load["video"], load["skip_first_frames"], load["frame_load_cap"]
        tw, th = prompt["2"]["inputs"]["width"], prompt["2"]["inputs"]["height"]
        probe = upscale.probe(src)
        cw, ch = upscale.center_crop(probe.width, probe.height, (tw, th))
        index = len(self.prompts) - 1
        keep = f"between(n\\,{skip}\\,{skip + cap - 1})"
        if self.drop == index:
            keep += f"*not(eq(n\\,{skip + cap // 2}))"
        out = self.tmp_path / f"seg{index}.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf",
                        f"select='{keep}',crop={cw}:{ch},scale={tw}:{th}:flags=lanczos",
                        "-vsync", "0", "-an", "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(out)], check=True)
        return str(out)


def gray_frames(path, size, crop=None):
    w, h = size
    vf = (f"crop={crop[0]}:{crop[1]}," if crop else "") + f"scale={w}:{h}:flags=area,format=gray"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.int16)


@pytest.fixture
def no_models(monkeypatch):
    monkeypatch.setattr(upscale, "ensure_models", lambda model_key, remaining: None)


@pytest.mark.parametrize("size, frames, segment", [
    ((848, 480), 50, 21),     # 3 segments with context
    ((480, 848), 37, 13),     # portrait, 3 segments, short last one
    ((848, 480), 30, None),   # one segment (RAM-sized)
])
def test_frames_fps_size_and_audio_are_kept(tmp_path, monkeypatch, no_models, size, frames, segment):
    src = make_clip(tmp_path / "src.mp4", size, frames)
    fake = FakeComfy(tmp_path)
    monkeypatch.setattr(upscale, "run_prompt", fake)
    out = str(tmp_path / "out.mp4")
    job = {"upscale_segment_frames": segment} if segment else {}

    result = upscale.run(src, "3b", out, lambda: 3600, job)

    source, video = upscale.probe(src), upscale.probe(out)
    target = upscale.target_size(*size)
    assert (video.frames, video.fps) == (source.frames, source.fps) == (frames, "25/1")
    assert (video.width, video.height) == target == result.out_size
    assert video.has_audio and upscale.audio_md5(out) == upscale.audio_md5(src)
    assert result.frames == frames and result.segments == len(fake.prompts)
    if segment:
        assert len(fake.prompts) == len(upscale.plan_segments(frames, segment)) > 1

    # every output frame is the source frame at the same index (not shifted, not repeated)
    small = upscale.CHECK_SIZES[target]
    a = gray_frames(src, small, upscale.center_crop(source.width, source.height, target))
    b = gray_frames(out, small)
    same = np.abs(a - b).mean(axis=(1, 2))
    shifted = np.abs(a[1:] - b[:-1]).mean(axis=(1, 2))
    assert same.mean() < 0.5 * shifted.mean(), (same.mean(), shifted.mean())


def test_a_frame_lost_inside_a_segment_fails(tmp_path, monkeypatch, no_models):
    src = make_clip(tmp_path / "src.mp4", (848, 480), 50)
    monkeypatch.setattr(upscale, "run_prompt", FakeComfy(tmp_path, drop=1))
    with pytest.raises(upscale.UpscaleError, match="segment 2/3 returned 36 frames, expected 37"):
        upscale.run(src, "3b", str(tmp_path / "out.mp4"), lambda: 3600, {"upscale_segment_frames": 21})
    assert not list(tmp_path.glob("seg*.mkv"))  # segment files removed on failure too


def test_output_with_another_frame_count_fails_the_check(tmp_path):
    src = make_clip(tmp_path / "src.mp4", (848, 480), 50)
    short = make_clip(tmp_path / "short.mp4", (1280, 720), 49)
    with pytest.raises(upscale.UpscaleError, match="output check failed: frames 49 != source 50"):
        upscale.verify(src, upscale.probe(src), short, (1280, 720))


def test_verify_catches_fps_size_and_audio(tmp_path):
    src = make_clip(tmp_path / "src.mp4", (848, 480), 25)
    source = upscale.probe(src)
    other_fps = make_clip(tmp_path / "fps.mp4", (1280, 720), 25, fps="24")
    with pytest.raises(upscale.UpscaleError, match="fps 24/1 != source 25/1"):
        upscale.verify(src, source, other_fps, (1280, 720))
    no_audio = make_clip(tmp_path / "silent.mp4", (1280, 720), 25, audio=False)
    with pytest.raises(upscale.UpscaleError, match="audio stream missing"):
        upscale.verify(src, source, no_audio, (1280, 720))
    with pytest.raises(upscale.UpscaleError, match="size 1280x720 != 720x1280"):
        upscale.verify(src, source, no_audio, (720, 1280))


def test_source_without_audio(tmp_path, monkeypatch, no_models):
    src = make_clip(tmp_path / "src.mp4", (848, 480), 20, audio=False)
    monkeypatch.setattr(upscale, "run_prompt", FakeComfy(tmp_path))
    out = str(tmp_path / "out.mp4")
    upscale.run(src, "3b", out, lambda: 3600, {})
    video = upscale.probe(out)
    assert video.frames == 20 and not video.has_audio


def test_logs(tmp_path, monkeypatch, no_models, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="worker_guard")
    src = make_clip(tmp_path / "src.mp4", (848, 480), 50)
    monkeypatch.setattr(upscale, "run_prompt", FakeComfy(tmp_path))
    upscale.run(src, "7b", str(tmp_path / "out.mp4"), lambda: 3600, {"upscale_segment_frames": 21})
    lines = caplog.messages
    assert any(m.startswith("[UPSCALE] segments: 50 frames -> 3 segment(s) of 21 frames (+8 context each side), "
                            "RAM available 56.0 GB, per frame 150 MB") for m in lines)
    assert any(m.startswith("[UPSCALE] model seedvr2_7b_fp16, input 848x480 -> output 1280x720, frames 50, time ")
               and " vram peak " in m for m in lines)
    assert any(m.startswith("[UPSCALE] RAM peak ") and "segment 21 frames, 3 segment(s)" in m for m in lines)
    assert any(m.startswith("[UPSCALE] frozen-frame check: 0 suspicious frames (of 50; by frame%4: 0/0/0/0")
               for m in lines), [m for m in lines if "frozen" in m]
    assert not list(tmp_path.glob("seg*.mkv"))  # segment files removed
