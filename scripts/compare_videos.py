#!/usr/bin/env python3
# ==============================================================================
# compare_videos.py - A/B quality check: compile OFF video vs compile ON video.
# PASS if the average SSIM >= 0.97 AND no single frame is below 0.90; also checks that
# both videos have the same frame count and resolution. Writes a side-by-side MP4
# (left = compile off, right = compile on, audio from the left) to check lip-sync by eye,
# and a per-frame SSIM CSV.
#
# Inputs can be .mp4 files or saved RunPod responses (.json with output.video base64,
# or a .txt with the raw base64).
#
# Usage (Windows PowerShell, needs ffmpeg: winget install Gyan.FFmpeg):
#   python scripts\compare_videos.py off.json on.json --out compare_out
# Exit code: 0 PASS, 1 FAIL, 2 could not compare.
# ==============================================================================
import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

AVG_MIN = 0.97
FRAME_MIN = 0.90


def find_ffmpeg(explicit=None):
    for candidate in (explicit, os.environ.get("FFMPEG"), shutil.which("ffmpeg")):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _extract_b64(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("video", "output"):
            if key in obj:
                found = _extract_b64(obj[key])
                if found:
                    return found
    return None


def to_mp4(path, workdir, label):
    """Returns a path to an .mp4 for `path` (.mp4 as-is; .json / .txt base64 decoded)."""
    if path.lower().endswith(".mp4"):
        return path
    # utf-8-sig: Windows PowerShell 5.1 Set-Content -Encoding UTF8 writes a BOM
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip()
    try:
        b64 = _extract_b64(json.loads(text))
    except ValueError:
        b64 = text
    if not b64:
        raise ValueError(f"{path}: no 'video' base64 found (expected output.video in the RunPod response)")
    out = os.path.join(workdir, f"{label}.mp4")
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64))
    return out


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stderr


def video_info(ffmpeg, path):
    """(frames, width, height) by decoding the first video stream."""
    rc, err = run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0", "-f", "null", "-"])
    if rc != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}:\n{err[-800:]}")
    frames = re.findall(r"frame=\s*(\d+)", err)
    size = re.search(r"Video: [^\n]*?(\d{2,5})x(\d{2,5})", err)
    return (int(frames[-1]) if frames else None,
            int(size.group(1)) if size else None, int(size.group(2)) if size else None)


def parse_ssim_log(text):
    """Per-frame SSIM ("All" column) from ffmpeg's ssim stats_file."""
    values = []
    for line in text.splitlines():
        m = re.search(r"All:([0-9.]+|inf)", line)
        if m:
            values.append(1.0 if m.group(1) == "inf" else float(m.group(1)))
    return values


def verdict(ssim, frames_a, frames_b, size_a, size_b):
    """Returns (passed, reasons)."""
    reasons = []
    if frames_a != frames_b:
        reasons.append(f"frame count differs: off {frames_a} vs on {frames_b}")
    if size_a != size_b:
        reasons.append(f"resolution differs: off {size_a} vs on {size_b}")
    if not ssim:
        reasons.append("no SSIM values computed")
        return False, reasons
    avg = sum(ssim) / len(ssim)
    low = [i for i, v in enumerate(ssim) if v < FRAME_MIN]
    if avg < AVG_MIN:
        reasons.append(f"average SSIM {avg:.4f} < {AVG_MIN}")
    if low:
        reasons.append(f"{len(low)} frame(s) below {FRAME_MIN} (first: frame {low[0]}, SSIM {ssim[low[0]]:.4f})")
    return not reasons, reasons


def compare(ffmpeg, off_mp4, on_mp4, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frames_a, *size_a = video_info(ffmpeg, off_mp4)
    frames_b, *size_b = video_info(ffmpeg, on_mp4)

    # stats_file is written relative to cwd, which avoids escaping Windows paths inside the filter string
    rc, err = run([ffmpeg, "-hide_banner", "-i", os.path.abspath(off_mp4), "-i", os.path.abspath(on_mp4),
                   "-lavfi", "[0:v][1:v]ssim=stats_file=ssim.log", "-f", "null", "-"], cwd=out_dir)
    log_path = os.path.join(out_dir, "ssim.log")
    if rc != 0 or not os.path.exists(log_path):
        raise RuntimeError(f"ffmpeg ssim failed:\n{err[-800:]}")
    with open(log_path, encoding="utf-8", errors="replace") as f:
        ssim = parse_ssim_log(f.read())
    with open(os.path.join(out_dir, "ssim_per_frame.csv"), "w", encoding="utf-8") as f:
        f.write("frame,ssim\n" + "".join(f"{i},{v:.6f}\n" for i, v in enumerate(ssim)))

    side_by_side = os.path.join(out_dir, "side_by_side_off_vs_on.mp4")
    rc, err = run([ffmpeg, "-hide_banner", "-y", "-i", off_mp4, "-i", on_mp4,
                   "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "0:a?",
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", side_by_side])
    if rc != 0:
        print(f"WARNING: side-by-side video failed:\n{err[-600:]}", file=sys.stderr)
        side_by_side = None

    passed, reasons = verdict(ssim, frames_a, frames_b, tuple(size_a), tuple(size_b))
    return {
        "passed": passed, "reasons": reasons, "ssim": ssim,
        "frames": (frames_a, frames_b), "size": (tuple(size_a), tuple(size_b)), "side_by_side": side_by_side,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="SSIM A/B check: compile off vs compile on")
    parser.add_argument("off", help="compile OFF result (.mp4, RunPod response .json, or base64 .txt)")
    parser.add_argument("on", help="compile ON result (.mp4, RunPod response .json, or base64 .txt)")
    parser.add_argument("--out", default="compare_out", help="output folder (default: compare_out)")
    parser.add_argument("--ffmpeg", help="path to ffmpeg.exe (default: FFMPEG env, PATH, imageio-ffmpeg)")
    args = parser.parse_args(argv)

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if not ffmpeg:
        print("ERROR: ffmpeg not found. Install it (winget install Gyan.FFmpeg) or pass --ffmpeg C:\\path\\ffmpeg.exe")
        return 2
    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = compare(ffmpeg, to_mp4(args.off, tmp, "compile_off"), to_mp4(args.on, tmp, "compile_on"), args.out)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 2

    ssim = result["ssim"]
    if ssim:
        avg, low = sum(ssim) / len(ssim), min(ssim)
        print(f"frames compared : {len(ssim)} (off {result['frames'][0]}, on {result['frames'][1]})")
        print(f"resolution      : off {result['size'][0]}, on {result['size'][1]}")
        print(f"average SSIM    : {avg:.4f} (needs >= {AVG_MIN})")
        print(f"lowest frame    : {low:.4f} at frame {ssim.index(low)} (needs >= {FRAME_MIN})")
    print(f"per-frame CSV   : {os.path.join(args.out, 'ssim_per_frame.csv')}")
    if result["side_by_side"]:
        print(f"side-by-side    : {result['side_by_side']} (left = compile off, right = compile on)")
    print("PASS" if result["passed"] else "FAIL: " + "; ".join(result["reasons"]))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
