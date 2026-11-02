"""v28 "stabilize": true - FFmpeg stabilization of the generated video (before the fast480 upscale).

vidstab two-pass when the installed ffmpeg has libvidstab (vidstabdetect + vidstabtransform), otherwise deshake.
One static zoom, just enough to hide the borders (optzoom=1), capped by maxshift / maxangle, so the crop stays
small; output size, frame count and fps unchanged; audio copied without re-encoding (packets checked by md5).
Re-encoded with libx264 crf 15 (the generation itself writes crf 19). Any failure returns an error string and the
caller keeps the unstabilized video.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time

from worker_guard import upscale

logger = logging.getLogger("worker_guard")

VIDSTAB, DESHAKE = "vidstab", "deshake"
CRF = 15
DETECT = "vidstabdetect=shakiness=5:accuracy=15:result={trf}"
TRANSFORM = "vidstabtransform=input={trf}:smoothing=10:optzoom=1:maxshift=24:maxangle=0.05:interpol=bicubic"
DESHAKE_FILTER = "deshake=edge=mirror"

_method = None


class StabilizeError(Exception):
    pass


def method():
    """vidstab if `ffmpeg -filters` lists vidstabdetect and vidstabtransform, else deshake (checked once)."""
    global _method
    if _method is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True,
                                 timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        _method = VIDSTAB if "vidstabdetect" in out and "vidstabtransform" in out else DESHAKE
    return _method


def _run(args, what, cwd=None, timeout=1800):
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, capture_output=True, text=True, timeout=timeout,
                          cwd=cwd)
    if proc.returncode != 0:
        raise StabilizeError(f"{what} failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")


def _encode_args(source_path, out_path, video_filter, source):
    return ["-i", os.path.abspath(source_path), "-vf", video_filter, "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-r", source.fps, "-vsync", "cfr",
            "-c:a", "copy", "-movflags", "+faststart", os.path.abspath(out_path)]


def run(source_path, out_path):
    """Writes the stabilized video to out_path. Returns (method, seconds, frames). Raises StabilizeError."""
    started = time.monotonic()
    chosen = method()
    try:
        source = upscale.probe(source_path)
        if chosen == VIDSTAB:
            # vidstab writes / reads its transforms file by a relative name in a private folder
            work = tempfile.mkdtemp(prefix="stab_")
            try:
                trf = "transforms.trf"
                _run(["-i", os.path.abspath(source_path), "-vf", DETECT.format(trf=trf), "-f", "null", "-"],
                     "vidstabdetect", cwd=work)
                _run(_encode_args(source_path, out_path, TRANSFORM.format(trf=trf), source), "vidstabtransform",
                     cwd=work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        else:
            _run(_encode_args(source_path, out_path, DESHAKE_FILTER, source), "deshake")
        out = upscale.verify(source_path, source, out_path, (source.width, source.height))
    except upscale.UpscaleError as e:
        raise StabilizeError(str(e))
    except subprocess.TimeoutExpired as e:
        raise StabilizeError(f"ffmpeg timed out after {e.timeout:.0f}s")
    seconds = time.monotonic() - started
    logger.info(f"[STAB] {chosen}, time {seconds:.1f}s, frames {out.frames}, {out.width}x{out.height}, crf {CRF}")
    return chosen, seconds, out.frames
