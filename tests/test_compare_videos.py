# ==============================================================================
# scripts/compare_videos.py: SSIM parsing, PASS/FAIL rule (avg >= 0.97 and no frame < 0.90),
# RunPod-response input decoding, and (when ffmpeg is installed) a real end-to-end run.
# ==============================================================================
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import compare_videos as cv  # noqa: E402

SIZE = (720, 480)


def test_parse_ssim_log():
    text = (
        "n:1 Y:0.991000 U:0.995000 V:0.994000 All:0.992000 (20.9)\n"
        "n:2 Y:1.000000 U:1.000000 V:1.000000 All:1.000000 (inf)\n"
        "garbage line\n"
        "n:3 Y:0.9 U:0.9 V:0.9 All:inf (inf)\n"
    )
    assert cv.parse_ssim_log(text) == [0.992, 1.0, 1.0]


@pytest.mark.parametrize("ssim, passed", [
    ([0.99] * 10, True),
    ([0.97] * 10, True),                    # exactly the average limit
    ([0.9699] * 10, False),                 # average just below 0.97
    ([1.0] * 99 + [0.90], True),            # one frame exactly at 0.90 is allowed
    ([1.0] * 99 + [0.8999], False),         # one frame below 0.90 fails even with a high average
    ([], False),
])
def test_verdict_thresholds(ssim, passed):
    ok, reasons = cv.verdict(ssim, len(ssim), len(ssim), SIZE, SIZE)
    assert ok is passed, reasons


def test_verdict_frame_count_and_size_mismatch():
    ok, reasons = cv.verdict([1.0] * 5, 5, 6, SIZE, SIZE)
    assert not ok and "frame count differs" in reasons[0]
    ok, reasons = cv.verdict([1.0] * 5, 5, 5, SIZE, (832, 480))
    assert not ok and "resolution differs" in reasons[0]


@pytest.mark.parametrize("wrap", [
    lambda b: json.dumps({"id": "x", "status": "COMPLETED", "output": {"video": b}}),
    lambda b: json.dumps({"video": b}),
    lambda b: b,
])
def test_to_mp4_decodes_runpod_responses(tmp_path, wrap):
    data = b"\x00\x00\x00\x18ftypmp42 fake"
    src = tmp_path / "resp.json"
    # Windows PowerShell 5.1 Set-Content -Encoding UTF8 writes a BOM
    src.write_bytes(b"\xef\xbb\xbf" + wrap(base64.b64encode(data).decode()).encode())
    out = cv.to_mp4(str(src), str(tmp_path), "off")
    assert Path(out).read_bytes() == data


def test_to_mp4_without_video_fails(tmp_path):
    src = tmp_path / "resp.json"
    src.write_text(json.dumps({"status": "FAILED", "error": "boom"}))
    with pytest.raises(ValueError, match="no 'video' base64"):
        cv.to_mp4(str(src), str(tmp_path), "off")


FFMPEG = cv.find_ffmpeg()


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_end_to_end_with_ffmpeg(tmp_path):
    def make(name, vf=None):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=1",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(tmp_path / name)]
        subprocess.run(cmd, check=True)
        return str(tmp_path / name)

    same = make("a.mp4")
    flipped = make("b.mp4", "hflip")
    assert cv.main([same, same, "--out", str(tmp_path / "out_same")]) == 0
    assert cv.main([same, flipped, "--out", str(tmp_path / "out_bad")]) == 1
    sbs = tmp_path / "out_same" / "side_by_side_off_vs_on.mp4"
    assert sbs.exists() and (tmp_path / "out_same" / "ssim_per_frame.csv").exists()
    frames, width, height = cv.video_info(FFMPEG, str(sbs))
    assert (width, height) == (320, 120)  # two 160x120 videos side by side
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(sbs)], capture_output=True, text=True)
    assert "Audio:" in probe.stderr  # audio kept for the lip-sync check


def test_missing_ffmpeg_exit_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cv, "find_ffmpeg", lambda explicit=None: None)
    assert cv.main(["a.mp4", "b.mp4", "--out", str(tmp_path)]) == 2
    assert "ffmpeg not found" in capsys.readouterr().out
