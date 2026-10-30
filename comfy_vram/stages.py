"""Per-stage VRAM peaks inside the ComfyUI process (on unless VRAM_STAGES=0).

A daemon thread reads torch.cuda.mem_get_info() every VRAM_STAGE_INTERVAL_MS (default 50) while a
node runs and keeps the highest device-level use per stage:

  vae_encode / vae_decode   inside WanVideoVAE.encode / decode (also the per-window VAE work that
                            runs inside node 128 in InfiniteTalk mode), and nodes WanVideoEncode / WanVideoDecode
  sampler                   the rest of node WanVideoSampler (transformer steps)
  other                     every other node

mem_get_info is device-wide (allocator pools and the CUDA context included), which is what an
out-of-memory error is about. Spikes shorter than the interval can be missed.
Nothing here touches a tensor, a stream or an argument: no GPU sync, no change to any result.
An OutOfMemoryError leaving WanVideoVAE.encode/decode is recorded with its node and window
(window = finished VAE decodes in that node + 1; each InfiniteTalk window ends with its decode).
GET /worker/vram returns the state of the last prompt.
"""
import functools
import logging
import os
import threading
import time

logger = logging.getLogger("worker_vram")

WRAPPER_DIR = "ComfyUI-WanVideoWrapper"
VAE_FILE = "wanvideo/wan_video_vae.py"
NODE_STAGES = {"WanVideoSampler": "sampler", "WanVideoDecode": "vae_decode", "WanVideoEncode": "vae_encode"}
DEFAULT_INTERVAL_MS = 50

# Set by the VAE wrappers in the executing thread, read by the sampling thread
_vae_kind = None
_internal_error_logged = False


def _safe(fn):
    """A bug here must never break a job."""
    global _internal_error_logged
    try:
        return fn()
    except Exception as e:
        if not _internal_error_logged:
            _internal_error_logged = True
            logger.warning(f"[VRAM] internal error (ignored): {type(e).__name__}: {e}")
        return None


def interval_sec():
    try:
        return max(10, int(os.environ.get("VRAM_STAGE_INTERVAL_MS", DEFAULT_INTERVAL_MS))) / 1000
    except ValueError:
        return DEFAULT_INTERVAL_MS / 1000


def is_oom(exc):
    return any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__)


class StageState:
    def __init__(self):
        self.lock = threading.Lock()
        self.vae_hooks = False
        self.reset({}, None, None)

    def reset(self, classes, total, baseline):
        with self.lock:
            self.classes = {str(k): v for k, v in (classes or {}).items()}
            self.total = total
            self.baseline = baseline
            self.peaks = {}
            self.node_peaks = {}
            self.decodes = {}
            self.samples = 0
            self.oom = None

    def stage_for(self, node_id, vae_kind):
        if vae_kind:
            return vae_kind
        if node_id is None:
            return None
        return NODE_STAGES.get(self.classes.get(str(node_id)), "other")

    def sample(self, used, total, node_id, vae_kind):
        stage = self.stage_for(node_id, vae_kind)
        if stage is None:
            return
        with self.lock:
            self.samples += 1
            self.total = total or self.total
            self.peaks[stage] = max(self.peaks.get(stage, 0), used)
            if node_id is not None:
                key = str(node_id)
                self.node_peaks[key] = max(self.node_peaks.get(key, 0), used)

    def decoded(self, node_id):
        if node_id is None:
            return
        with self.lock:
            key = str(node_id)
            self.decodes[key] = self.decodes.get(key, 0) + 1

    def record_oom(self, kind, node_id):
        with self.lock:
            if self.oom is None:
                key = None if node_id is None else str(node_id)
                self.oom = {"stage": kind, "node_id": key, "node_type": self.classes.get(key),
                            "window": self.decodes.get(key, 0) + 1}

    def snapshot(self):
        with self.lock:
            return {
                "enabled": True, "vae_hooks": self.vae_hooks, "interval_ms": round(interval_sec() * 1000),
                "total": self.total, "baseline": self.baseline, "samples": self.samples,
                "stages": dict(self.peaks), "nodes": dict(self.node_peaks), "decodes": dict(self.decodes),
                "oom": dict(self.oom) if self.oom else None,
            }


STATE = StageState()


def _current_node():
    try:
        from server import PromptServer
        return PromptServer.instance.last_node_id
    except Exception:
        return None


def _wrap_vae(kind, orig):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        global _vae_kind
        outer = _vae_kind
        _vae_kind = kind
        try:
            out = orig(*args, **kwargs)
        except BaseException as e:
            if is_oom(e):
                _safe(lambda: STATE.record_oom(kind, _current_node()))
            raise
        finally:
            _vae_kind = outer
        if kind == "vae_decode" and outer is None:
            _safe(lambda: STATE.decoded(_current_node()))
        return out
    return wrapper


def find_vae_class(modules, wrapper_dir=WRAPPER_DIR):
    suffix = f"/{wrapper_dir}/{VAE_FILE}"
    hits = {id(m): m for m in list(modules.values())
            if str(getattr(m, "__file__", None) or "").replace("\\", "/").endswith(suffix)}
    if len(hits) != 1:
        return None, f"{len(hits)} loaded module(s) for {wrapper_dir}/{VAE_FILE}"
    cls = getattr(next(iter(hits.values())), "WanVideoVAE", None)
    if cls is None or not all(callable(getattr(cls, n, None)) for n in ("encode", "decode")):
        return None, "WanVideoVAE.encode/decode missing"
    return cls, None


_vae_patched = None  # None: not tried yet


def patch_vae(modules=None):
    global _vae_patched
    if _vae_patched is not None:
        return _vae_patched
    import sys
    cls, problem = find_vae_class(sys.modules if modules is None else modules)
    if cls is None:
        _vae_patched = False
        logger.warning(f"[VRAM] VAE stages off ({problem}); VAE work inside the sampler counts as 'sampler'")
        return False
    cls.encode = _wrap_vae("vae_encode", cls.encode)
    cls.decode = _wrap_vae("vae_decode", cls.decode)
    _vae_patched = STATE.vae_hooks = True
    return True


class Sampler(threading.Thread):
    def __init__(self, mem_get_info, interval):
        super().__init__(name="vram-stages", daemon=True)
        self.mem_get_info = mem_get_info
        self.interval = interval

    def run(self):
        while True:
            time.sleep(self.interval)
            node, kind = _current_node(), _vae_kind
            if node is None and kind is None:
                continue
            _safe(lambda: self._sample(node, kind))

    def _sample(self, node, kind):
        free, total = self.mem_get_info()
        STATE.sample(total - free, total, node, kind)


_sampler = None
_mem_get_info = None


def _device_mem_get_info():
    import torch
    import comfy.model_management as mm
    device = mm.get_torch_device()
    return functools.partial(torch.cuda.mem_get_info, device)


def on_prompt(json_data):
    def work():
        global _sampler, _mem_get_info
        patch_vae()
        if _mem_get_info is None:
            _mem_get_info = _device_mem_get_info()
        prompt = (json_data or {}).get("prompt") or {}
        classes = {k: v.get("class_type") for k, v in prompt.items() if isinstance(v, dict)}
        free, total = _mem_get_info()
        STATE.reset(classes, total, total - free)
        if _sampler is None:
            _sampler = Sampler(_mem_get_info, interval_sec())
            _sampler.start()
            logger.info(f"[VRAM] stage sampler on: every {interval_sec() * 1000:.0f} ms, "
                        f"VAE stages {'on' if STATE.vae_hooks else 'off'}")
    _safe(work)
    return json_data


def register():
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/worker/vram")
    async def worker_vram(request):
        return web.json_response(STATE.snapshot())

    PromptServer.instance.add_on_prompt_handler(on_prompt)
    logger.info("[VRAM] per-stage VRAM peaks: GET /worker/vram (sampler starts with the first prompt)")
