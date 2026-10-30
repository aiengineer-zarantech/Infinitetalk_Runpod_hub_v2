# ==============================================================================
# v26: the frozen-frame check (ComfyUI issue #16591: near-frozen frames every 4th frame after SeedVR2) finds
# frames where the source moves but the upscaled video repeats the frame before. It only logs.
# ==============================================================================
import shutil
import subprocess

import numpy as np
import pytest

from worker_guard import upscale


def test_suspicious_frames():
    source = [2.0, 2.0, 2.0, 2.0, 0.3, 2.0]
    result = [2.1, 0.1, 1.9, 0.4, 0.0, 0.6]
    # frame i+1 is suspicious when the source moves (> 0.5) and the result moves < 25% of that
    assert upscale.suspicious_frames(source, result) == [2, 4]


def test_line_format():
    assert upscale.frozen_frame_line([4, 8, 9, 12], 50) == (
        "[UPSCALE] frozen-frame check: 4 suspicious frames (of 50; by frame%4: 3/1/0/0; first: 4, 8, 9, 12)")
    assert upscale.frozen_frame_line([], 375) == (
        "[UPSCALE] frozen-frame check: 0 suspicious frames (of 375; by frame%4: 0/0/0/0; first: -)")


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="needs ffmpeg/ffprobe (present in the worker image)")


def _source(path, frames=50):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=848x480:rate=25",
                    "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 480, 848, 3)


def _upscaled(path, frames):
    """Like the upscale output: center crop to 16:9 and 1280x720, lossless."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "848x480",
                           "-r", "25", "-i", "-", "-vf", "crop=848:477,scale=1280:720", "-c:v", "ffv1",
                           "-pix_fmt", "yuv444p", str(path)], input=frames.tobytes(), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()[-500:]
    return str(path)


@needs_ffmpeg
def test_every_4th_frame_frozen_is_found(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="worker_guard")
    src = tmp_path / "src.mp4"
    frames = _source(src).copy()
    for i in range(4, len(frames), 4):
        frames[i] = frames[i - 1]          # the #16591 pattern
    out = _upscaled(tmp_path / "out.mkv", frames)
    found = upscale.frozen_check(str(src), upscale.probe(str(src)), out, (1280, 720))
    assert found == list(range(4, 50, 4))
    assert "[UPSCALE] frozen-frame check: 12 suspicious frames (of 50; by frame%4: 12/0/0/0; first: 4, 8, 12, 16, " \
           "20, 24, 28, 32, 36, 40)" in caplog.messages


@needs_ffmpeg
def test_clean_upscale_has_no_suspicious_frames(tmp_path):
    src = tmp_path / "src.mp4"
    out = _upscaled(tmp_path / "out.mkv", _source(src))
    assert upscale.frozen_check(str(src), upscale.probe(str(src)), out, (1280, 720)) == []
