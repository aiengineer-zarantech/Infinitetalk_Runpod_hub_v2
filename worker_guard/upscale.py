"""SeedVR2 720p upscale stage (v26), using the native SeedVR2 nodes of ComfyUI core (d07dff45).

Runs after the InfiniteTalk generation ("mode": "fast480") or alone ("mode": "upscale_only"):
- The video is always upscaled in time segments sized from the container RAM (sysmem.read_ram). Each segment
  loads CONTEXT_FRAMES extra frames on both sides so SeedVR2 sees the motion across the cut; only its core
  frames are kept. One ComfyUI prompt per segment (Upscale_SeedVR2.json), written as a lossless ffv1 file.
- ffmpeg keeps the core frames, encodes H.264 (crf 19, yuv420p, like the generation output) and copies the
  source audio unchanged (-c:a copy).
- Frame count, frame rate, size and the audio packets must equal the source, else UpscaleError.
- The frozen-frame check (ComfyUI issue #16591: near-frozen frames every 4th frame) only logs.
An out-of-memory error retries only that segment with smaller temporal chunks and VAE tiles; the InfiniteTalk
generation is never re-run for it.
"""
import base64
import binascii
import copy
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from fractions import Fraction

from worker_guard import comfy_api, sysmem

logger = logging.getLogger("worker_guard")

GB = 1024 ** 3
MB = 1024 ** 2

WORKFLOW_FILE = "Upscale_SeedVR2.json"
MODEL_FILES = {"3b": "seedvr2_3b_fp16.safetensors", "7b": "seedvr2_7b_fp16.safetensors"}
VAE_FILE = "seedvr2_ema_vae_fp16.safetensors"
LOADERS = {"UNETLoader": "unet_name", "VAELoader": "vae_name"}

# Workflow nodes the stage fills in
LOAD_NODE, RESIZE_NODE, ENCODE_NODE, CHUNK_NODE, UNET_NODE = "1", "2", "5", "6", "7"
DECODE_NODE, POST_NODE, SAVE_NODE = "11", "12", "13"

LANDSCAPE, PORTRAIT = (1280, 720), (720, 1280)
MAX_ASPECT_ERROR = 0.03

CONTEXT_FRAMES = 8
SEGMENT_MIN, SEGMENT_MAX = 33, 401
SEGMENT_UNKNOWN_RAM = 97
RAM_MARGIN_GB = 4.0
DEFAULT_RAM_PER_FRAME_MB = 150.0
COLOR_METHODS = ("lab", "wavelet", "adain", "none")

# Frozen-frame check: grayscale at this size; a frame is suspicious when the source moves more than
# MOTION_FLOOR gray levels on average but the result moves less than FROZEN_RATIO of that
CHECK_SIZES = {LANDSCAPE: (320, 180), PORTRAIT: (180, 320)}
MOTION_FLOOR = 0.5
FROZEN_RATIO = 0.25

POLL_SEC = 1.0
COMFY_UNREACHABLE_SEC = 60
TIME_SAFETY = 1.3

CLIENT_ID = f"upscale-{uuid.uuid4()}"
INPUT_DIR = "/ComfyUI/input"
OUTPUT_DIR = "/ComfyUI/output"

# Swappable in tests
_now = time.monotonic
_sleep = time.sleep


class UpscaleError(Exception):
    """The upscale could not produce a correct video. fast480 then returns the 480p video."""


class SegmentError(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__(f"ComfyUI {details.get('type', 'execution_error')} at node {details.get('node_id')} "
                         f"({details.get('node_type')}): {details.get('exception_type')}: "
                         f"{str(details.get('exception_message') or '').strip()[:300]}")

    @property
    def is_oom(self):
        text = f"{self.details.get('exception_type')} {self.details.get('exception_message')}"
        return "OutOfMemoryError" in text or "out of memory" in text.lower()


@dataclass(frozen=True)
class Attempt:
    """Settings of one try of a segment. chunk None = SeedVR2TemporalChunk auto (sized from free VRAM)."""
    chunk: int = None
    tile: int = 512
    overlap: int = 128
    temporal: int = 64
    temporal_overlap: int = 8


FIRST_ATTEMPT = Attempt()
OOM_RETRIES = (
    Attempt(chunk=21),
    Attempt(chunk=13, tile=384, overlap=96, temporal=32),
    Attempt(chunk=9, tile=256, overlap=64, temporal=16),
)


@dataclass(frozen=True)
class Probe:
    frames: int
    fps: str  # exact r_frame_rate, e.g. "25/1"
    width: int
    height: int
    has_audio: bool

    @property
    def fps_value(self):
        return float(Fraction(self.fps))


@dataclass(frozen=True)
class Segment:
    index: int
    start: int       # first core frame (kept)
    end: int         # one past the last core frame
    load_start: int  # first frame loaded (core minus context)
    load_end: int

    @property
    def load_count(self):
        return self.load_end - self.load_start


@dataclass
class Result:
    path: str
    model_file: str
    in_size: tuple
    out_size: tuple
    frames: int
    fps: str
    seconds: float
    vram_peak: int
    ram_peak: int
    ram_limit: int
    segment_frames: int
    segments: int
    suspicious: list = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ ffmpeg

def _run(cmd, timeout=1800):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise UpscaleError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")
    return proc.stdout


def probe(path):
    """Exact frame count (decoded), r_frame_rate, size and whether there is an audio stream."""
    out = _run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_type,width,height,r_frame_rate,nb_read_frames", "-of", "json", path])
    streams = json.loads(out).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise UpscaleError(f"no video stream in {os.path.basename(path)}")
    try:
        frames = int(video["nb_read_frames"])
        Fraction(video["r_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise UpscaleError(f"cannot read frames/fps of {os.path.basename(path)}: {e}")
    return Probe(frames, video["r_frame_rate"], int(video["width"]), int(video["height"]),
                 any(s.get("codec_type") == "audio" for s in streams))


def count_packets(path):
    """Frames in a lossless segment file without decoding it (ffv1: one packet per frame)."""
    out = _run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                "stream=nb_read_packets", "-of", "json", path])
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise UpscaleError(f"no video stream in segment file {os.path.basename(path)}")
    return int(streams[0]["nb_read_packets"])


def audio_md5(path):
    """md5 of the first audio stream's packets as stored (no decoding), or None without audio."""
    out = _run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0?", "-c", "copy", "-f", "md5", "-"])
    out = out.strip()
    return out or None


def target_size(width, height):
    """1280x720 for landscape, 720x1280 for portrait. Raises if the aspect is far from 16:9 / 9:16."""
    target = LANDSCAPE if width >= height else PORTRAIT
    error = abs((width / height) / (target[0] / target[1]) - 1)
    if error > MAX_ASPECT_ERROR:
        raise UpscaleError(f"input {width}x{height} is {error:.1%} away from {target[0]}:{target[1]} "
                           f"(max {MAX_ASPECT_ERROR:.0%}); upscale supports 16:9 and 9:16 only")
    return target


def center_crop(width, height, target):
    """The source crop (w, h) with the exact target aspect, centered: what ImageScale crop 'center' uses."""
    tw, th = target
    if width * th > height * tw:
        return round(height * tw / th), height
    return width, round(width * th / tw)


# ------------------------------------------------------------------------------------------------ segments

def ram_per_frame_mb():
    raw = os.environ.get("UPSCALE_RAM_PER_FRAME_MB", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_RAM_PER_FRAME_MB
    except ValueError:
        logger.warning(f"[UPSCALE] invalid UPSCALE_RAM_PER_FRAME_MB={raw!r}, using {DEFAULT_RAM_PER_FRAME_MB:.0f}")
        value = DEFAULT_RAM_PER_FRAME_MB
    return max(1.0, value)


def _round_4n1(n):
    return max(1, ((int(n) - 1) // 4) * 4 + 1)


def segment_frames(available_bytes, per_frame_mb, override=None):
    """Core frames per segment (4n+1). The RAM must hold the core plus 2 x CONTEXT_FRAMES loaded frames."""
    if override:
        return min(SEGMENT_MAX, max(5, _round_4n1(override)))
    if available_bytes is None:
        return SEGMENT_UNKNOWN_RAM
    budget = available_bytes - RAM_MARGIN_GB * GB
    frames = int(budget / (per_frame_mb * MB)) - 2 * CONTEXT_FRAMES
    return min(SEGMENT_MAX, max(SEGMENT_MIN, _round_4n1(frames)))


def plan_segments(total, size, context=CONTEXT_FRAMES):
    segments, start = [], 0
    while start < total:
        end = min(start + size, total)
        segments.append(Segment(len(segments), start, end, max(0, start - context), min(total, end + context)))
        start = end
    return segments


# ------------------------------------------------------------------------------------------------ workflow

def load_template(workflow_dir=None):
    workflow_dir = workflow_dir or os.environ.get("WORKFLOW_DIR", "/")
    with open(os.path.join(workflow_dir, WORKFLOW_FILE), encoding="utf-8") as f:
        return json.load(f)


def build_prompt(template, source_path, segment, target, model_file, attempt, fps_value, color, prefix):
    prompt = copy.deepcopy(template)
    prompt[LOAD_NODE]["inputs"].update(video=source_path, skip_first_frames=segment.load_start,
                                       frame_load_cap=segment.load_count)
    prompt[RESIZE_NODE]["inputs"].update(width=target[0], height=target[1])
    prompt[UNET_NODE]["inputs"]["unet_name"] = model_file
    for node in (ENCODE_NODE, DECODE_NODE):
        prompt[node]["inputs"].update(tile_size=attempt.tile, overlap=attempt.overlap,
                                      temporal_size=attempt.temporal, temporal_overlap=attempt.temporal_overlap)
    chunk = prompt[CHUNK_NODE]["inputs"]
    if attempt.chunk:
        chunk["chunking_mode"] = "manual"
        chunk["chunking_mode.frames_per_chunk"] = attempt.chunk
    else:
        chunk["chunking_mode"] = "auto"
        chunk.pop("chunking_mode.frames_per_chunk", None)
    prompt[POST_NODE]["inputs"]["color_correction_method"] = color
    prompt[SAVE_NODE]["inputs"].update(frame_rate=fps_value, filename_prefix=prefix)
    return prompt


def _execution_error(entry):
    for message in (entry.get("status") or {}).get("messages") or []:
        if isinstance(message, (list, tuple)) and len(message) == 2 and message[0] in (
                "execution_error", "execution_interrupted"):
            return {"type": message[0], **(message[1] or {})}
    return None


def run_prompt(prompt, remaining):
    """Queues one segment and waits for it. Returns the path of the written segment file."""
    try:
        answer = comfy_api.queue(prompt, CLIENT_ID)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise UpscaleError(f"ComfyUI rejected the upscale workflow (HTTP {e.code}): {body[:1500]}")
    if answer.get("node_errors"):
        raise UpscaleError(f"ComfyUI rejected the upscale workflow: {json.dumps(answer['node_errors'])[:1500]}")
    prompt_id = answer["prompt_id"]
    unreachable_since = None
    while True:
        if remaining() <= 0:
            try:
                comfy_api.interrupt(prompt_id)
            except Exception:
                pass
            raise UpscaleError("job time limit reached during the upscale")
        try:
            entry = comfy_api.history(prompt_id).get(prompt_id)
            unreachable_since = None
        except Exception as e:
            unreachable_since = unreachable_since or _now()
            if _now() - unreachable_since > COMFY_UNREACHABLE_SEC:
                raise UpscaleError(f"ComfyUI unreachable for {COMFY_UNREACHABLE_SEC}s during the upscale: "
                                   f"{type(e).__name__}: {e}")
            entry = None
        if entry:
            error = _execution_error(entry)
            if error:
                raise SegmentError(error)
            status = entry.get("status") or {}
            if status.get("completed") or status.get("status_str") == "success":
                files = (entry.get("outputs") or {}).get(SAVE_NODE, {}).get("gifs") or []
                if not files or not os.path.exists(files[0].get("fullpath", "")):
                    raise UpscaleError(f"segment prompt {prompt_id} finished without a video file")
                return files[0]["fullpath"]
        _sleep(POLL_SEC)


# ------------------------------------------------------------------------------------------------ assembly

def assemble(segment_files, segments, source_path, source, out_path):
    """Core frames of every segment, in order, at the source frame rate; source audio copied unchanged."""
    rate = Fraction(source.fps)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for path in segment_files:
        cmd += ["-i", path]
    cmd += ["-i", source_path]
    parts = []
    for i, seg in enumerate(segments):
        a, b = seg.start - seg.load_start, seg.end - seg.load_start
        parts.append(f"[{i}:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]")
    joined = "".join(f"[v{i}]" for i in range(len(segments)))
    parts.append(f"{joined}concat=n={len(segments)}:v=1:a=0,"
                 f"setpts=N*{rate.denominator}/{rate.numerator}/TB[v]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", f"{len(segments)}:a:0?",
            "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-r", source.fps, "-vsync", "cfr",
            "-c:a", "copy", "-movflags", "+faststart", out_path]
    _run(cmd, timeout=3600)


def verify(source_path, source, out_path, target):
    """Raises UpscaleError unless frames, fps, size and audio packets match. Returns the output Probe."""
    out = probe(out_path)
    problems = []
    if out.frames != source.frames:
        problems.append(f"frames {out.frames} != source {source.frames}")
    if out.fps != source.fps:
        problems.append(f"fps {out.fps} != source {source.fps}")
    if (out.width, out.height) != tuple(target):
        problems.append(f"size {out.width}x{out.height} != {target[0]}x{target[1]}")
    if out.has_audio != source.has_audio:
        problems.append(f"audio stream {'added' if out.has_audio else 'missing'}")
    elif source.has_audio and audio_md5(out_path) != audio_md5(source_path):
        problems.append("audio packets differ from the source")
    if problems:
        raise UpscaleError("output check failed: " + "; ".join(problems))
    return out


# ------------------------------------------------------------------------------------------------ frozen frames

def motion(path, size, crop=None):
    """Mean absolute gray-level difference between each frame and the one before (len = frames - 1)."""
    import numpy as np

    w, h = size
    vf = (f"crop={crop[0]}:{crop[1]}," if crop else "") + f"scale={w}:{h}:flags=area,format=gray"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    values, previous, n = [], None, w * h
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            if previous is not None:
                values.append(float(np.abs(frame - previous).mean()))
            previous = frame
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        if proc.wait() != 0:
            raise UpscaleError(f"ffmpeg failed reading {os.path.basename(path)}: {stderr.strip()[-300:]}")
    return values


def suspicious_frames(source_motion, result_motion, floor=MOTION_FLOOR, ratio=FROZEN_RATIO):
    """Frame indices where the source moves but the result (nearly) repeats the frame before."""
    return [i + 1 for i, (s, r) in enumerate(zip(source_motion, result_motion)) if s > floor and r < ratio * s]


def frozen_frame_line(frames, total):
    by_mod = [sum(1 for f in frames if f % 4 == k) for k in range(4)]
    first = ", ".join(str(f) for f in frames[:10]) or "-"
    return (f"[UPSCALE] frozen-frame check: {len(frames)} suspicious frames (of {total}; "
            f"by frame%4: {'/'.join(str(c) for c in by_mod)}; first: {first})")


def frozen_check(source_path, source, out_path, target):
    size = CHECK_SIZES[tuple(target)]
    src = motion(source_path, size, center_crop(source.width, source.height, target))
    res = motion(out_path, size)
    if len(src) != len(res):
        logger.warning(f"[UPSCALE] frozen-frame check: {len(src) + 1} source vs {len(res) + 1} result frames")
    frames = suspicious_frames(src, res)
    logger.info(frozen_frame_line(frames, len(src) + 1))
    return frames


# ------------------------------------------------------------------------------------------------ peaks

class Peaks(threading.Thread):
    """Samples ComfyUI VRAM and container RAM every second while the upscale runs."""

    def __init__(self, interval=POLL_SEC):
        super().__init__(name="upscale-peaks", daemon=True)
        self.interval = interval
        self.vram = 0
        self.ram = 0
        self.ram_limit = None
        self._stop_event = threading.Event()

    def sample(self):
        try:
            device = comfy_api.first_device(comfy_api.system_stats(timeout=5))
            if device:
                self.vram = max(self.vram, device["vram_used"])
        except Exception:
            pass
        try:
            ram = sysmem.read_ram()
            self.ram_limit = ram.get("limit")
            if ram.get("used") is not None:
                self.ram = max(self.ram, ram["used"])
        except Exception:
            pass

    def note_stage_peaks(self):
        """Adds the 50 ms per-stage peaks of the last prompt (zz_worker_vram), when available."""
        try:
            stages = (comfy_api.vram_stages() or {}).get("stages") or {}
            values = [v for v in stages.values() if isinstance(v, (int, float))]
            if values:
                self.vram = max(self.vram, int(max(values)))
        except Exception:
            pass

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.sample()

    def stop(self):
        self._stop_event.set()
        self.sample()


# ------------------------------------------------------------------------------------------------ models

def ensure_models(model_key, remaining):
    """Volume -> local disk -> ComfyUI for the model and the VAE (download once if missing). Raises UpscaleError."""
    import setup_models

    names = [MODEL_FILES[model_key], VAE_FILE]
    outcome = {}

    def work():
        try:
            outcome["info"] = [setup_models.ensure_upscale_model(name) for name in names]
        except Exception as e:
            outcome["error"] = f"{type(e).__name__}: {e}"

    started = _now()
    worker = threading.Thread(target=work, name="upscale-model-copy", daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, remaining()))
    if worker.is_alive():
        raise UpscaleError(f"preparing {', '.join(names)} did not finish within the job time limit")
    if "error" in outcome:
        raise UpscaleError(f"model files not ready: {outcome['error']}")
    for node_class, input_name in LOADERS.items():
        name = names[0] if node_class == "UNETLoader" else names[1]
        try:
            spec = comfy_api.object_info(node_class)[node_class]["input"]["required"][input_name]
            options = spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
        except Exception as e:
            raise UpscaleError(f"/object_info/{node_class} failed: {type(e).__name__}: {e}")
        if name not in options:
            raise UpscaleError(f"ComfyUI {node_class} does not list {name}")
    copied = [i["model"] for i in outcome["info"] if i.get("copied")]
    logger.info(f"[UPSCALE] models ready in {_now() - started:.1f}s: {', '.join(names)}"
                + (f" (copied now: {', '.join(copied)})" if copied else ""))


# ------------------------------------------------------------------------------------------------ input / output

def save_input_video(job_input, task_id, input_dir=INPUT_DIR):
    """upscale_only input -> local file path."""
    if job_input.get("video_path"):
        path = job_input["video_path"]
        if not os.path.isfile(path):
            raise UpscaleError(f"video_path {path} does not exist")
        return path
    os.makedirs(input_dir, exist_ok=True)
    path = os.path.join(input_dir, f"{task_id}_upscale_input.mp4")
    if job_input.get("video_base64"):
        data = job_input["video_base64"]
        if "," in data[:100] and data.startswith("data:"):
            data = data.split(",", 1)[1]
        try:
            raw = base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError) as e:
            raise UpscaleError(f"video_base64 is not valid base64: {e}")
        with open(path, "wb") as f:
            f.write(raw)
        return path
    url = job_input.get("video_url")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(path, "wb") as f:
            while True:
                chunk = response.read(8 * MB)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise UpscaleError(f"video_url download failed: {type(e).__name__}: {e}")
    return path


# ------------------------------------------------------------------------------------------------ main

def run(source_path, model_key, out_path, remaining, job_input=None):
    """Upscales source_path to out_path. Returns Result; raises UpscaleError (or another exception on a bug)."""
    job_input = job_input or {}
    started = _now()
    model_file = MODEL_FILES[model_key]
    color = str(job_input.get("upscale_color_correction", "lab")).strip().lower()
    if color not in COLOR_METHODS:
        raise UpscaleError(f"upscale_color_correction must be one of {list(COLOR_METHODS)}")

    source = probe(source_path)
    target = target_size(source.width, source.height)
    logger.info(f"[UPSCALE] source {source.width}x{source.height}, {source.frames} frames at {source.fps} fps, "
                f"audio {'yes' if source.has_audio else 'no'} -> {target[0]}x{target[1]}, model {model_file}")
    ensure_models(model_key, remaining)

    ram = sysmem.read_ram()
    per_frame = ram_per_frame_mb()
    size = segment_frames(ram.get("available"), per_frame, job_input.get("upscale_segment_frames"))
    segments = plan_segments(source.frames, size)
    available = "n/a" if ram.get("available") is None else f"{ram['available'] / GB:.1f} GB"
    logger.info(f"[UPSCALE] segments: {source.frames} frames -> {len(segments)} segment(s) of {size} frames "
                f"(+{CONTEXT_FRAMES} context each side), RAM available {available}, per frame {per_frame:.0f} MB"
                + (" (segment size from upscale_segment_frames)" if job_input.get("upscale_segment_frames") else ""))
    if ram.get("available") is None:
        logger.warning(f"[UPSCALE] RAM available unknown: using {SEGMENT_UNKNOWN_RAM}-frame segments")

    first = FIRST_ATTEMPT
    if job_input.get("upscale_chunk_frames"):
        first = Attempt(chunk=_round_4n1(job_input["upscale_chunk_frames"]))

    template = load_template()
    peaks = Peaks()
    peaks.start()
    segment_files = []
    spent_per_frame = None
    try:
        for seg in segments:
            if spent_per_frame is not None:
                need = spent_per_frame * seg.load_count * TIME_SAFETY
                if remaining() < need:
                    raise UpscaleError(f"not enough time for segment {seg.index + 1}/{len(segments)}: needs "
                                       f"~{need:.0f}s, {remaining():.0f}s left")
            attempts = (first,) + OOM_RETRIES
            for k, attempt in enumerate(attempts):
                prompt = build_prompt(template, source_path, seg, target, model_file, attempt,
                                      source.fps_value, color, f"SeedVR2_{uuid.uuid4().hex[:8]}_seg{seg.index}")
                seg_started = _now()
                try:
                    path = run_prompt(prompt, remaining)
                except SegmentError as e:
                    if e.is_oom and k + 1 < len(attempts):
                        nxt = attempts[k + 1]
                        logger.warning(f"[UPSCALE] OOM in segment {seg.index + 1}/{len(segments)}: retry "
                                       f"{k + 1}/{len(OOM_RETRIES)} with chunk {nxt.chunk} frames, tile {nxt.tile}, "
                                       f"temporal {nxt.temporal}")
                        try:
                            comfy_api.free(unload_models=False, free_memory=True)
                        except Exception:
                            pass
                        continue
                    raise UpscaleError(f"segment {seg.index + 1}/{len(segments)}: {e}")
                segment_files.append(path)
                written = count_packets(path)
                if written != seg.load_count:
                    # a lost frame would shift every later frame of the segment against the audio
                    raise UpscaleError(f"segment {seg.index + 1}/{len(segments)} returned {written} frames, "
                                       f"expected {seg.load_count}")
                peaks.note_stage_peaks()
                elapsed = _now() - seg_started
                spent_per_frame = elapsed / max(1, seg.load_count)
                logger.info(f"[UPSCALE] segment {seg.index + 1}/{len(segments)}: frames {seg.start}-{seg.end - 1} "
                            f"(loaded {seg.load_start}-{seg.load_end - 1}) in {elapsed:.1f}s"
                            + (f" after {k} OOM retr{'y' if k == 1 else 'ies'}" if k else ""))
                break

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        assemble(segment_files, segments, source_path, source, out_path)
        verify(source_path, source, out_path, target)
    finally:
        peaks.stop()
        for path in segment_files:
            try:
                os.remove(path)
            except OSError:
                pass

    suspicious = []
    try:
        suspicious = frozen_check(source_path, source, out_path, target)
    except Exception as e:  # the check only logs; it never fails the job
        logger.warning(f"[UPSCALE] frozen-frame check: not run ({type(e).__name__}: {e})")

    seconds = _now() - started
    name = model_file.rsplit(".", 1)[0]
    logger.info(f"[UPSCALE] model {name}, input {source.width}x{source.height} -> output {target[0]}x{target[1]}, "
                f"frames {source.frames}, time {seconds:.1f}s, vram peak {peaks.vram / GB:.1f} GB")
    limit = "n/a" if peaks.ram_limit is None else f"{peaks.ram_limit / GB:.1f} GB"
    logger.info(f"[UPSCALE] RAM peak {peaks.ram / GB:.1f} GB (limit {limit}), segment {size} frames, "
                f"{len(segments)} segment(s)")
    return Result(out_path, model_file, (source.width, source.height), target, source.frames, source.fps, seconds,
                  peaks.vram, peaks.ram, peaks.ram_limit, size, len(segments), suspicious)
