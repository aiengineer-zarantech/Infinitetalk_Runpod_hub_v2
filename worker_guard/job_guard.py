"""Per-job GPU guard, lazy model preparation, ComfyUI watchdog and job timing.

handler.py calls install() once, right before runpod.serverless.start():

    handler, get_videos = install_job_guard(handler, get_videos)

The wrapped handler checks ComfyUI's device before every job (except "cmd" diagnostics),
copies the Multi model when a multi job needs it, and logs one [JOB] line per job.
The wrapped get_videos logs progress, polls VRAM and enforces JOB_TIMEOUT_SEC while the
workflow runs. Set WORKER_GUARD_ENABLED=0 to get both functions back unchanged.
"""
import functools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass

from worker_guard import accel, comfy_api, offload
from worker_guard.node_timer import NodeTimer
from worker_guard.speed import SpeedTracker

logger = logging.getLogger("worker_guard")

MULTI_MODEL = "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"
MULTI_LOADER_CLASS = "MultiTalkModelLoader"
MODEL_FOLDER = "diffusion_models"

# Prompt nodes the handler fills in: max_frame, width, height
FRAMES_NODE, WIDTH_NODE, HEIGHT_NODE = "270", "245", "246"

DEFAULT_VRAM_NEAR_ZERO_MB = 2048
# VHS_VideoCombine encodes with ffmpeg on CPU, so low VRAM there is expected
DEFAULT_EXEMPT_NODES = "VHS_VideoCombine"

# Swappable in tests
_now = time.monotonic
_sleep = time.sleep

# The job currently running (the handler runs with concurrency 1)
_current = None


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[GUARD] invalid {name}={raw!r}, using {default}")
        return float(default)


@dataclass(frozen=True)
class Settings:
    job_timeout_sec: float
    vram_poll_sec: float
    vram_sample_sec: float
    vram_near_zero_mb: float
    vram_idle_timeout_sec: float
    progress_log_sec: float
    interrupt_grace_sec: float
    gpu_check_retry_sec: float
    free_wait_sec: float
    exempt_nodes: frozenset

    @classmethod
    def from_env(cls):
        exempt = os.environ.get("VRAM_WATCHDOG_EXEMPT_NODES", DEFAULT_EXEMPT_NODES)
        return cls(
            job_timeout_sec=_env_float("JOB_TIMEOUT_SEC", 3600),
            vram_poll_sec=_env_float("VRAM_POLL_SEC", 30),
            vram_sample_sec=_env_float("VRAM_SAMPLE_SEC", 5),
            vram_near_zero_mb=_env_float("VRAM_NEAR_ZERO_MB", DEFAULT_VRAM_NEAR_ZERO_MB),
            vram_idle_timeout_sec=_env_float("VRAM_IDLE_TIMEOUT_SEC", 180),
            progress_log_sec=_env_float("PROGRESS_LOG_SEC", 5),
            interrupt_grace_sec=_env_float("INTERRUPT_GRACE_SEC", 60),
            gpu_check_retry_sec=_env_float("JOB_GPU_CHECK_RETRY_SEC", 60),
            free_wait_sec=_env_float("OOM_FREE_WAIT_SEC", 30),
            exempt_nodes=frozenset(n.strip() for n in exempt.split(",") if n.strip()),
        )


def _node_value(prompt, node_id):
    try:
        return prompt[node_id]["inputs"]["value"]
    except (KeyError, TypeError):
        return None


def _gb(nbytes):
    return nbytes / (1024 ** 3)


def gpu_name(device):
    """'cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync' -> 'NVIDIA GeForce RTX 4090'."""
    name = (device or {}).get("name") or "unknown GPU"
    if name.startswith("cuda:") and " " in name:
        name = name.split(" ", 1)[1]
    return name.rsplit(" : ", 1)[0].strip()


class JobContext:
    def __init__(self, settings, task):
        self.settings = settings
        self.task = task
        self.started = _now()
        self.deadline = self.started + settings.job_timeout_sec
        self.frames = self.width = self.height = None
        self.node_classes = {}
        self.prompt_id = None
        self.executing = False
        self.current_node = None
        self.last_progress_log = None
        self.trip_kind = None
        self.trip_message = None
        self.tripped_at = None
        self.refresh_worker = False
        self.device = None
        self.offload = None
        self.oom_retry = False
        self.peak_vram = 0
        self.benchmark = False
        self.compile_on = False
        self.compile_note = "off"
        self.compile_fallback = False
        self.fp8_fast = False
        self.speed = None
        self.nodes = None
        self.lock = threading.Lock()

    def note_vram(self, used_bytes):
        self.peak_vram = max(self.peak_vram, used_bytes)

    def swapping(self):
        return bool(self.offload and self.offload.blocks_to_swap > 0)

    def remaining(self):
        return self.deadline - _now()

    def record_prompt(self, prompt):
        self.frames = _node_value(prompt, FRAMES_NODE)
        self.width = _node_value(prompt, WIDTH_NODE)
        self.height = _node_value(prompt, HEIGHT_NODE)
        self.node_classes = {
            str(node_id): node.get("class_type") for node_id, node in prompt.items() if isinstance(node, dict)
        }

    def current_class(self):
        return self.node_classes.get(str(self.current_node)) if self.current_node is not None else None

    def trip(self, kind, message, refresh):
        """Records the first watchdog trip only. Returns True if this call tripped it."""
        with self.lock:
            if self.trip_kind:
                return False
            self.trip_kind = kind
            self.trip_message = message
            self.tripped_at = _now()
            self.refresh_worker = refresh
            return True


class WsProxy:
    """Passes every websocket message through unchanged and watches progress/executing messages."""

    def __init__(self, ws, ctx):
        self._ws = ws
        self._ctx = ctx

    def recv(self):
        out = self._ws.recv()
        if isinstance(out, str):
            try:
                self._observe(json.loads(out))
            except Exception:
                pass
        return out

    def __getattr__(self, name):
        return getattr(self._ws, name)

    def _observe(self, message):
        ctx = self._ctx
        mtype = message.get("type")
        data = message.get("data") or {}
        prompt_id = data.get("prompt_id")
        if prompt_id and ctx.prompt_id and prompt_id != ctx.prompt_id:
            return
        if mtype == "execution_start":
            ctx.prompt_id = prompt_id or ctx.prompt_id
            ctx.executing = True
        elif mtype == "executing":
            node = data.get("node")
            if ctx.speed is not None:
                ctx.speed.on_executing(node, _now())
            if ctx.nodes is not None:
                ctx.nodes.on_executing(node, _now())
            if node is None:
                ctx.current_node = None
                ctx.executing = False
            else:
                ctx.prompt_id = ctx.prompt_id or prompt_id
                ctx.current_node = node
                ctx.executing = True
        elif mtype == "execution_cached":
            if ctx.nodes is not None:
                ctx.nodes.on_cached(data.get("nodes"))
        elif mtype == "progress":
            ctx.executing = True
            now = _now()
            if ctx.speed is not None:
                ctx.speed.on_progress(data.get("node", ctx.current_node), data.get("value"), data.get("max"), now)
            if ctx.last_progress_log is None or now - ctx.last_progress_log >= ctx.settings.progress_log_sec:
                ctx.last_progress_log = now
                node = data.get("node", ctx.current_node)
                node_class = ctx.node_classes.get(str(node), "?")
                logger.info(
                    f"[PROGRESS] node {node} ({node_class}) {data.get('value')}/{data.get('max')} "
                    f"(elapsed {now - ctx.started:.0f}s)"
                )


def _close_ws(ws):
    for name in ("shutdown", "close"):
        fn = getattr(ws, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception as e:
                logger.warning(f"[WATCHDOG] websocket {name}() failed: {e}")


class Watchdog(threading.Thread):
    """Checks the job deadline every second, samples /system_stats every VRAM_SAMPLE_SEC
    (peak VRAM + near-zero rule) and logs VRAM every VRAM_POLL_SEC."""

    def __init__(self, ctx, ws):
        super().__init__(name="comfy-watchdog", daemon=True)
        self.ctx = ctx
        self.ws = ws
        self.next_sample = _now() + min(ctx.settings.vram_sample_sec, ctx.settings.vram_poll_sec)
        self.next_poll = _now() + ctx.settings.vram_poll_sec
        self.low_vram_since = None
        self.ws_closed = False
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(1.0):
            try:
                self.tick()
            except Exception as e:
                logger.warning(f"[WATCHDOG] check failed: {e}")

    def stop(self):
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5)

    def tick(self):
        ctx, s = self.ctx, self.ctx.settings
        now = _now()
        if ctx.trip_kind:
            # Interrupt sent but ComfyUI is still running: the worker is stuck
            if not self.ws_closed and now - ctx.tripped_at >= s.interrupt_grace_sec:
                with ctx.lock:
                    ctx.refresh_worker = True
                logger.error(
                    f"[WATCHDOG] ComfyUI did not stop within INTERRUPT_GRACE_SEC={s.interrupt_grace_sec:.0f}s; "
                    "closing websocket and refreshing worker"
                )
                self.ws_closed = True
                _close_ws(self.ws)
            return
        if now >= ctx.deadline:
            self._trip(
                "timeout",
                f"[WATCHDOG] job exceeded JOB_TIMEOUT_SEC={s.job_timeout_sec:.0f}s; workflow interrupted",
                refresh=False,
            )
            return
        if now >= self.next_sample:
            self.next_sample = now + min(s.vram_sample_sec, s.vram_poll_sec)
            log_it = now >= self.next_poll
            if log_it:
                self.next_poll = now + s.vram_poll_sec
            self._poll_vram(now, log_it)

    def _exempt(self, node_class):
        s = self.ctx.settings
        if node_class in s.exempt_nodes:
            return True
        # With blocks swapped the handler sets load_device=offload_device, so loaders fill CPU RAM, not VRAM
        return self.ctx.swapping() and bool(node_class) and node_class.endswith("Loader")

    def _poll_vram(self, now, log_it=True):
        ctx, s = self.ctx, self.ctx.settings
        try:
            device = comfy_api.first_device(comfy_api.system_stats(timeout=10))
        except Exception as e:
            if log_it:
                logger.warning(f"[WATCHDOG] /system_stats failed: {type(e).__name__}: {e}")
            return
        if device is None:
            if log_it:
                logger.warning("[WATCHDOG] /system_stats reported no devices")
            return
        ctx.note_vram(device["vram_used"])
        node_class = ctx.current_class()
        if log_it:
            logger.info(
                f"[WATCHDOG] VRAM used {_gb(device['vram_used']):.1f} / {_gb(device['vram_total']):.1f} GB, "
                f"node {ctx.current_node} ({node_class or '-'})"
            )
        if device["type"] != "cuda":
            self._trip(
                "device",
                f"[WATCHDOG] ComfyUI device changed to '{device['type']}' during the job; workflow interrupted",
                refresh=True,
            )
            return
        used_mb = device["vram_used"] / (1024 ** 2)
        if not ctx.executing or used_mb >= s.vram_near_zero_mb or self._exempt(node_class):
            self.low_vram_since = None
            return
        if self.low_vram_since is None:
            self.low_vram_since = now
        elif now - self.low_vram_since > s.vram_idle_timeout_sec:
            self._trip(
                "vram_idle",
                f"[WATCHDOG] VRAM near zero (< {s.vram_near_zero_mb:.0f} MB) for "
                f"{now - self.low_vram_since:.0f}s while executing node {ctx.current_node} "
                f"({node_class or '-'}); workflow interrupted",
                refresh=True,
            )

    def _trip(self, kind, message, refresh):
        if not self.ctx.trip(kind, message, refresh):
            return
        logger.error(message)
        try:
            comfy_api.interrupt(self.ctx.prompt_id)
            logger.warning("[WATCHDOG] sent /interrupt to ComfyUI")
        except Exception as e:
            logger.error(f"[WATCHDOG] /interrupt failed: {type(e).__name__}: {e}")


def _check_comfy_gpu(settings):
    """Returns (None, device) if ComfyUI runs on cuda, else (error result for the job, device or None)."""
    deadline = _now() + settings.gpu_check_retry_sec
    while True:
        try:
            device = comfy_api.first_device(comfy_api.system_stats(timeout=10))
            break
        except Exception as e:
            if _now() >= deadline:
                message = (
                    f"[GPU] ComfyUI /system_stats unreachable for {settings.gpu_check_retry_sec:.0f}s "
                    f"({type(e).__name__}: {e}). Workflow not queued."
                )
                logger.error(message)
                return {"error": message, "refresh_worker": True}, None
            _sleep(2)
    if device is None or device["type"] != "cuda":
        device_type = device["type"] if device else "none"
        name = device["name"] if device else "no devices reported"
        message = f"[GPU] ComfyUI device is '{device_type}' ({name}), expected cuda. Workflow not queued."
        logger.error(message)
        return {"error": message, "refresh_worker": True}, device
    logger.info(
        f"[GPU] job check OK: cuda, {device['name']}, VRAM used "
        f"{_gb(device['vram_used']):.1f} / {_gb(device['vram_total']):.1f} GB"
    )
    return None, device


def _loader_options(info, node_class):
    spec = info[node_class]["input"]["required"]["model"]
    if isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict) and "options" in spec[1]:
        return spec[1]["options"]
    raise KeyError(f"unexpected {node_class} model input format")


def _comfy_lists_model(model_name, attempts=3):
    """Asks ComfyUI for its model list (this also refreshes its mtime-based file list cache)."""
    detail = ""
    for attempt in range(attempts):
        try:
            options = _loader_options(comfy_api.object_info(MULTI_LOADER_CLASS), MULTI_LOADER_CLASS)
            if model_name in options:
                return True, f"/object_info/{MULTI_LOADER_CLASS}"
            detail = f"/object_info/{MULTI_LOADER_CLASS} lists {len(options)} model(s) without it"
        except Exception as e:
            detail = f"/object_info/{MULTI_LOADER_CLASS} failed ({type(e).__name__}: {e})"
            try:
                if model_name in (comfy_api.list_models(MODEL_FOLDER) or []):
                    return True, f"/models/{MODEL_FOLDER}"
                detail += f"; /models/{MODEL_FOLDER} does not list it"
            except Exception as e2:
                detail += f"; /models/{MODEL_FOLDER} failed ({type(e2).__name__}: {e2})"
        if attempt < attempts - 1:
            _sleep(1)
    return False, detail


def _ensure_multi_model(ctx):
    """Copies and links the Multi model if needed. Returns None when ComfyUI sees it, else an error result."""
    import setup_models

    outcome = {}

    def work():
        try:
            outcome["info"] = setup_models.ensure_model(MULTI_MODEL)
        except Exception as e:
            outcome["error"] = f"{type(e).__name__}: {e}"

    logger.info(f"[MODELS] multi job: preparing {MULTI_MODEL}")
    worker = threading.Thread(target=work, name="lazy-model-copy", daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, ctx.remaining()))
    if worker.is_alive():
        message = f"[MODELS] copying {MULTI_MODEL} did not finish within JOB_TIMEOUT_SEC. Workflow not queued."
        logger.error(message)
        return {"error": message, "refresh_worker": True}
    if "error" in outcome:
        message = f"[MODELS] failed to prepare {MULTI_MODEL}: {outcome['error']}. Workflow not queued."
        logger.error(message)
        return {"error": message}

    info = outcome["info"]
    seen, how = _comfy_lists_model(MULTI_MODEL)
    if not seen:
        message = f"[MODELS] ComfyUI does not list {MULTI_MODEL} ({how}). Workflow not queued."
        logger.error(message)
        return {"error": message}
    if info.get("copied"):
        copied = (
            f"copied {info['size_mb']:.1f} MB in {info.get('copy_seconds', 0):.1f}s "
            f"({info.get('speed_mb_s', 0):.1f} MB/s)"
        )
    else:
        copied = "already local"
    logger.info(
        f"[MODELS] {MULTI_MODEL} ready in {info.get('seconds', 0):.1f}s: {copied}; "
        f"{info['dest_path']} -> {info['local_path']}; ComfyUI lists it via {how}"
    )
    return None


def _apply_trip(ctx, result):
    """Turns the result of a job the watchdog interrupted into a clear error."""
    if not ctx.trip_kind:
        return result
    out = dict(result) if isinstance(result, dict) else {}
    finished = not out.get("error") and (out.get("video") or out.get("video_path"))
    if not finished:
        original = out.get("error")
        out["error"] = ctx.trip_message + (f" | ComfyUI: {original}" if original else "")
    if ctx.refresh_worker:
        out["refresh_worker"] = True
    return out


def _free_vram(ctx):
    """Asks ComfyUI to unload models, then waits (bounded) until VRAM actually drops.
    /free is applied asynchronously, so queueing right away could retry with the models still loaded."""
    s = ctx.settings
    try:
        comfy_api.free()
    except Exception as e:
        logger.warning(f"[OFFLOAD] /free failed: {type(e).__name__}: {e}")
    deadline = _now() + s.free_wait_sec
    used = None
    while True:
        try:
            device = comfy_api.first_device(comfy_api.system_stats(timeout=10))
            if device is not None:
                used = device["vram_used"]
                if used / (1024 ** 2) < s.vram_near_zero_mb:
                    logger.info(f"[OFFLOAD] /free done: VRAM used {_gb(used):.1f} GB")
                    return
        except Exception:
            pass
        if _now() >= deadline:
            shown = "unknown" if used is None else f"{_gb(used):.1f} GB"
            logger.warning(f"[OFFLOAD] VRAM used still {shown} {s.free_wait_sec:.0f}s after /free; retrying anyway")
            return
        _sleep(1)


def _offload_summary(ctx):
    d = ctx.offload
    width = ctx.width if ctx.width is not None else d.width
    height = ctx.height if ctx.height is not None else d.height
    return {
        "gpu": gpu_name(ctx.device),
        "vram_gb": round(d.vram_gb, 1),
        "width": width,
        "height": height,
        "frames": ctx.frames,
        "blocks_to_swap": d.blocks_to_swap,
        "force_offload": d.force_offload,
        "mode": d.mode,
        "oom_retry": ctx.oom_retry,
    }


def _retry_failed(ctx, result):
    """Error result after the OOM retry also failed."""
    info = _offload_summary(ctx)
    what = "out of memory again" if offload.is_oom(result) else "failed again"
    original = result.get("error") if isinstance(result, dict) else None
    message = (
        f"[OFFLOAD] {what} after OOM retry: GPU {info['gpu']}, VRAM {info['vram_gb']:.1f} GB, "
        f"{info['width']}x{info['height']}, frames {_fmt(info['frames'])}, "
        f"blocks_to_swap {info['blocks_to_swap']}, force_offload {info['force_offload']}"
        + (f" | ComfyUI: {original}" if original else "")
    )
    logger.error(message)
    out = {"error": message, "offload": info, "refresh_worker": True}
    if isinstance(result, dict) and "details" in result:
        out["details"] = result["details"]
    return out


def _fmt(value):
    return "n/a" if value is None else value


def _log_job_line(ctx, result):
    elapsed = _now() - ctx.started
    if ctx.task == "tts":
        what = "task tts"
    else:
        resolution = "n/a" if ctx.width is None or ctx.height is None else f"{ctx.width}x{ctx.height}"
        what = f"frames {_fmt(ctx.frames)}, resolution {resolution}"
        if ctx.device:
            what += (
                f", gpu {gpu_name(ctx.device)}, vram {_gb(ctx.device['vram_total']):.1f} GB, "
                f"peak {_gb(ctx.peak_vram):.1f} GB"
            )
        if ctx.offload:
            what += (
                f", blocks_to_swap {ctx.offload.blocks_to_swap}, force_offload {ctx.offload.force_offload}, "
                f"mode {ctx.offload.mode}"
            )
        what += f", oom_retry {'yes' if ctx.oom_retry else 'no'}"
        what += f", torch_compile {'on' if ctx.compile_on else 'off'}"
        sp = ctx.speed
        overhead = sp.first_step_overhead() if sp else None
        if ctx.compile_on:
            what += f", compile ~{overhead:.1f}s" if overhead is not None else ", compile n/a"
        if ctx.compile_fallback:
            what += ", compile_fallback yes"
        what += f", fp8_fast {'on' if ctx.fp8_fast else 'off'}"
        avg = sp.avg_s_per_it() if sp else None
        sampling = sp.sampling_sec() if sp else None
        what += f", avg {avg:.2f} s/it" if avg is not None else ", avg n/a s/it"
        what += f", sampling {sampling:.1f}s" if sampling is not None else ", sampling n/a"
        if ctx.benchmark:
            what += f", benchmark {accel.BENCHMARK_SECONDS}s audio"
    if isinstance(result, dict) and not result.get("error"):
        logger.info(f"[JOB] done in {elapsed:.1f}s ({what})")
    else:
        error = result.get("error") if isinstance(result, dict) else "exception raised"
        logger.error(f"[JOB] failed in {elapsed:.1f}s ({what}): {str(error)[:300]}")


def final_offload_check(prompt):
    """Called by handler.queue_prompt right before /prompt, also with WORKER_GUARD_ENABLED=0.
    With the guard on, the workflow must match this job's offload decision (OOM retry included);
    without it, only the 480p hard rule is enforced, using the VRAM ComfyUI reports."""
    ctx = _current
    if ctx is not None and ctx.offload is not None:
        d = ctx.offload
        return offload.enforce_final(prompt, d.vram_gb, (d.blocks_to_swap, d.force_offload), ctx.oom_retry)
    vram_gb = None
    try:
        device = comfy_api.first_device(comfy_api.system_stats())
        if device and device["vram_total"]:
            vram_gb = _gb(device["vram_total"])
    except Exception as e:
        logger.warning(f"[OFFLOAD] /system_stats failed before queueing ({type(e).__name__}: {e}); assuming >= "
                       f"{offload.FULL_GPU_MIN_VRAM_GB} GB")
    return offload.enforce_final(prompt, vram_gb)


def install(handler, get_videos):
    """Returns (guarded_handler, monitored_get_videos)."""
    if os.environ.get("WORKER_GUARD_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.warning("[GUARD] WORKER_GUARD_ENABLED=0: job guard, watchdog and job timing are disabled")
        return handler, get_videos

    s = Settings.from_env()
    logger.info(
        f"[GUARD] enabled: JOB_TIMEOUT_SEC={s.job_timeout_sec:.0f}, VRAM_POLL_SEC={s.vram_poll_sec:.0f}, "
        f"VRAM_NEAR_ZERO_MB={s.vram_near_zero_mb:.0f}, VRAM_IDLE_TIMEOUT_SEC={s.vram_idle_timeout_sec:.0f}, "
        f"exempt nodes={sorted(s.exempt_nodes)}"
    )

    @functools.wraps(handler)
    def guarded_handler(job):
        global _current
        job_input = job.get("input", {}) or {}
        task = "tts" if job_input.get("task") == "tts" else "video"
        if task != "tts" and "cmd" in job_input:
            return handler(job)

        ctx = JobContext(Settings.from_env(), task)
        _current = ctx
        result = None

        def run(run_job):
            try:
                out = handler(run_job)
            except Exception:
                if not ctx.trip_kind:
                    raise
                out = None
            return _apply_trip(ctx, out)

        try:
            result, ctx.device = _check_comfy_gpu(ctx.settings)
            if result is not None:
                return result
            ctx.note_vram(ctx.device["vram_used"])
            reject = offload.vram_reject(ctx.device["vram_total"], gpu_name(ctx.device))
            if reject:
                logger.error(reject)
                result = {"error": reject, "refresh_worker": True}
                return result
            if task == "tts":
                result = run(job)
                return result

            decision, reject = offload.decide(job_input, ctx.device["vram_total"], gpu_name(ctx.device))
            if reject:
                logger.error(reject)
                result = {"error": reject, "refresh_worker": True}
                return result
            ctx.offload = decision
            if decision.mode == "auto":
                logger.info(
                    f"[OFFLOAD] auto: VRAM {decision.vram_gb:.1f} GB, {decision.width}x{decision.height} -> "
                    f"blocks_to_swap {decision.blocks_to_swap}, force_offload {decision.force_offload}"
                )
            else:
                logger.info(
                    f"[OFFLOAD] manual: blocks_to_swap {decision.blocks_to_swap}, "
                    f"force_offload {decision.force_offload} (as sent)"
                )

            ctx.benchmark = bool(job_input.get("benchmark"))
            if ctx.benchmark:
                try:
                    job_input = accel.benchmark_input(job_input)
                except Exception as e:
                    result = {"error": f"[BENCHMARK] {type(e).__name__}: {e}"}
                    logger.error(result["error"])
                    return result
                logger.info(f"[BENCHMARK] audio trimmed to the first {accel.BENCHMARK_SECONDS}s")
            ctx.compile_on, ctx.compile_note = accel.compile_decision(job_input, ctx.benchmark)
            ctx.fp8_fast = accel.fp8_decision(job_input, ctx.benchmark)
            logger.info(f"[COMPILE] torch.compile {ctx.compile_note}; fp8_fast {'on' if ctx.fp8_fast else 'off'}")

            if job_input.get("person_count") == "multi":
                result = _ensure_multi_model(ctx)
                if result is not None:
                    return result

            result = run({**job, "input": offload.apply(job_input, decision)})
            if ctx.compile_on and not ctx.trip_kind and accel.is_compile_error(result):
                details = result.get("details", {})
                reason = f"{details.get('exception_type')}: {str(details.get('exception_message') or '').strip()[:300]}"
                logger.error(f"[COMPILE] failed: {reason}; re-running once with the original workflow")
                accel.disable_compile("compile failed earlier on this worker")
                ctx.compile_on, ctx.compile_note, ctx.compile_fallback = False, "off (fallback)", True
                _free_vram(ctx)
                result = run({**job, "input": offload.apply(job_input, decision)})
            if decision.mode == "auto" and not ctx.trip_kind and offload.is_oom(result):
                old = decision.blocks_to_swap
                decision.blocks_to_swap, decision.force_offload = offload.emergency_settings(old, decision.force_offload)
                ctx.oom_retry = True
                logger.warning(
                    f"[OFFLOAD] emergency swap after OOM: blocks_to_swap {old} -> {decision.blocks_to_swap}, "
                    f"force_offload {decision.force_offload}"
                )
                _free_vram(ctx)
                result = run({**job, "input": offload.apply(job_input, decision)})
                if not ctx.trip_kind and isinstance(result, dict) and result.get("error"):
                    result = _retry_failed(ctx, result)
            return result
        finally:
            _current = None
            _log_job_line(ctx, result)

    @functools.wraps(get_videos)
    def monitored_get_videos(ws, prompt, input_type="image", person_count="single"):
        ctx = _current
        if ctx is None:
            return get_videos(ws, prompt, input_type, person_count)
        ctx.record_prompt(prompt)
        if ctx.remaining() <= 0:
            message = (
                f"[WATCHDOG] job exceeded JOB_TIMEOUT_SEC={ctx.settings.job_timeout_sec:.0f}s "
                "before the workflow was queued"
            )
            ctx.trip("timeout", message, refresh=False)
            logger.error(message)
            raise TimeoutError(message)
        if ctx.compile_on:
            name = accel.workflow_name(input_type, person_count)
            try:
                additions = accel.compiled_additions(input_type, person_count)
                if additions is None:
                    ctx.compile_note = f"off (no compiled variant of {name})"
                else:
                    accel.apply_additions(prompt, additions)
            except Exception as e:
                additions = None
                ctx.compile_note = f"off (compiled workflow unusable: {type(e).__name__}: {e})"
            if additions is None:
                ctx.compile_on = False
                logger.warning(f"[COMPILE] {ctx.compile_note}; queueing the original workflow")
            else:
                logger.info(f"[COMPILE] using {accel.COMPILED_WORKFLOWS[name]}: added {additions[1]}")
        if ctx.fp8_fast:
            logger.warning(accel.apply_fp8_fast(prompt))
        ctx.speed = SpeedTracker(accel.steps_per_window(prompt), accel.SAMPLER_NODE)
        ctx.nodes = NodeTimer() if accel.env_flag("PROFILE", False) else None
        watchdog = Watchdog(ctx, ws)
        watchdog.start()
        try:
            return get_videos(WsProxy(ws, ctx), prompt, input_type, person_count)
        finally:
            watchdog.stop()
            if ctx.nodes is not None:
                ctx.nodes.log(ctx.node_classes)

    return guarded_handler, monitored_get_videos
