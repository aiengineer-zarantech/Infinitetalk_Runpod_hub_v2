"""MODEL_LOCK (default on): load the InfiniteTalk model once per worker and never reload it.

ComfyUI keeps a loaded model only while every input of the loader chain (WanVideoModelLoader and everything
upstream: LoRA, block swap, compile settings) stays identical, and it unloads ALL models after any OOM. So:

1. Locked config (locked_model_config.json): one workflow, compile setting, precision, quantization, LoRA,
   attention, blocks_to_swap and offload flags for every job. Job fields that would change them are ignored
   (logged); multi-person, video input and larger-than-locked sizes are rejected with a clear message.
2. Resident check: the loader-chain signature of the last successful job is kept. Before queueing:
   same signature -> "MODEL REUSED"; none yet -> "MODEL LOADED (cold start)"; different -> error, the job is
   refused (never a second copy, never a silent reload).
3. VRAM pre-flight: predicted peak for this frame count + non-model memory now must leave headroom_gb free.
   If not: clear non-model memory only (gc + empty_cache through the zz_worker_mem node, never /free, which
   also drops ComfyUI's cache and with it the model), then tiled final VAE decode, then fail fast with
   refresh_worker (no blocks_to_swap escalation, no retry with other settings on this worker).
MODEL_LOCK=0 restores the previous behaviour (per-job offload decision, OOM ladder, compile fallback,
benchmark overrides): use it on a separate diagnostics endpoint only.
"""
import hashlib
import json
import logging
import os
import threading

logger = logging.getLogger("worker_guard")

GB = 1024 ** 3
LOADER, LORA, BLOCKSWAP, SAMPLER, DECODE = "122", "138", "134", "128", "130"
FRAMES_NODE, WIDTH_NODE, HEIGHT_NODE = "270", "245", "246"

# Job fields that change the model or load another model: ignored under the lock
MODEL_FIELDS = ("blocks_to_swap", "force_offload", "load_device", "base_precision", "quantization",
                "attention_mode", "torch_compile", "fp8_fast", "merge_loras", "lora", "lora_strength",
                "vae_tiling", "offload_img_emb", "offload_txt_emb", "benchmark", "mode", "upscale",
                "upscale_model", "workflow")
# Job fields that cannot run under the lock at all (another workflow / model / VRAM budget)
_DEFAULT_PATHS = ("/locked_model_config.json",
                  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locked_model_config.json"))


def enabled():
    return os.environ.get("MODEL_LOCK", "1").strip().lower() not in ("0", "false", "no", "off")


_config = None


def config():
    """The locked config (cached). MODEL_LOCK_CONFIG overrides the path."""
    global _config
    if _config is None:
        paths = [os.environ["MODEL_LOCK_CONFIG"]] if os.environ.get("MODEL_LOCK_CONFIG") else list(_DEFAULT_PATHS)
        for path in paths:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    _config = json.load(f)
                _config["_path"] = path
                break
        else:
            raise FileNotFoundError(f"locked model config not found in {paths}")
    return _config


def reset_config():
    global _config
    _config = None


def model_settings(cfg=None):
    cfg = cfg or config()
    return {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("vram", "max_width", "max_height")}


def config_hash(cfg=None):
    raw = json.dumps(model_settings(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------- job input
def sanitize(job_input, cfg=None):
    """Returns (job_input for the handler, ignored notes, error or None). The returned input carries the locked
    offload values, so the handler writes the same node inputs for every job."""
    cfg = cfg or config()
    if job_input.get("person_count", "single") != "single":
        return None, [], ("[MODEL_LOCK] multi-person jobs load another model; this worker is locked to "
                          "single-person I2V (send them to a MODEL_LOCK=0 endpoint)")
    if job_input.get("input_type", "image") != "image":
        return None, [], ("[MODEL_LOCK] video input uses another workflow; this worker is locked to "
                          "image input (I2V)")
    try:
        width, height = int(job_input.get("width", cfg["max_width"])), int(job_input.get("height", cfg["max_height"]))
    except (TypeError, ValueError):
        return None, [], "[MODEL_LOCK] width / height must be integers"
    if width * height > cfg["max_width"] * cfg["max_height"]:
        return None, [], (f"[MODEL_LOCK] {width}x{height} is larger than the locked {cfg['max_width']}x"
                          f"{cfg['max_height']}: its VRAM need was not measured (no blocks_to_swap change allowed)")
    out = dict(job_input)
    ignored = []
    locked = {"blocks_to_swap": cfg["blocks_to_swap"], "force_offload": cfg["force_offload"],
              "load_device": cfg["load_device"], "base_precision": cfg["base_precision"]}
    for key in MODEL_FIELDS:
        if key in out:
            value = out.pop(key)
            if key not in locked or value != locked[key]:
                ignored.append(f"{key}={value!r}")
    out.update(locked)
    return out, ignored, None


# ------------------------------------------------------------------------------------------- the prompt
def enforce(prompt, cfg=None):
    """Writes the locked values into the loader chain. Returns the corrections (empty when the handler
    already wrote them, which is the normal case)."""
    cfg = cfg or config()
    offload = cfg["blocks_to_swap"] > 0
    want = {
        LOADER: {"base_precision": cfg["base_precision"], "quantization": cfg["quantization"],
                 "load_device": cfg["load_device"], "attention_mode": cfg["attention_mode"]},
        LORA: {"lora": cfg["lora"], "strength": cfg["lora_strength"], "merge_loras": cfg["merge_loras"]},
        BLOCKSWAP: {"blocks_to_swap": cfg["blocks_to_swap"], "offload_img_emb": cfg["offload_img_emb"] or offload,
                    "offload_txt_emb": cfg["offload_txt_emb"] or offload},
        SAMPLER: {"force_offload": cfg["force_offload"]},
    }
    fixes = []
    for node_id, values in want.items():
        inputs = (prompt.get(node_id) or {}).setdefault("inputs", {}) if node_id in prompt else None
        if inputs is None:
            fixes.append(f"node {node_id} missing")
            continue
        for key, value in values.items():
            if inputs.get(key) != value:
                fixes.append(f"{node_id}.{key} {inputs.get(key)!r}->{value!r}")
                inputs[key] = value
    return fixes


def frames(prompt):
    try:
        return int(prompt[FRAMES_NODE]["inputs"]["value"])
    except (KeyError, TypeError, ValueError):
        return None


def set_tiled_decode(prompt):
    inputs = (prompt.get(DECODE) or {}).get("inputs")
    if inputs is None:
        return False
    inputs["enable_vae_tiling"] = True
    return True


def predict_peak_gb(n_frames, tiled, cfg=None):
    v = (cfg or config())["vram"]
    decode = v["tiled_decode_gb"] if tiled else v["decode_gb_per_frame"] * (n_frames or 0)
    return v["sampler_peak_gb"] + decode


def preflight(prompt, device, resident_idle_bytes, clear_cache, read_device, cfg=None):
    """VRAM check before queueing. device / read_device(): {"vram_total", "vram_free"} in bytes.
    clear_cache(): frees non-model memory in ComfyUI, returns True if it ran. Returns a dict with
    ok, message, tiled, projected_peak_gb, total_gb, cleared."""
    cfg = cfg or config()
    v = cfg["vram"]
    n = frames(prompt)
    total = device["vram_total"] / GB
    report = {"frames": n, "total_gb": round(total, 2), "cleared": False, "tiled": False}

    def projected(dev, tiled):
        used = (dev["vram_total"] - dev["vram_free"]) / GB
        baseline = resident_idle_bytes / GB if resident_idle_bytes else v["cold_idle_gb"]
        extra = max(0.0, used - baseline)          # memory that is neither the model nor the CUDA context
        return predict_peak_gb(n, tiled, cfg) + extra, used, extra

    peak, used, extra = projected(device, False)
    report.update(vram_used_start_gb=round(used, 2), non_model_gb=round(extra, 2))
    if extra > v["garbage_clear_above_gb"] or total - peak < v["headroom_gb"]:
        if clear_cache():
            report["cleared"] = True
            device = read_device()
            peak, used, extra = projected(device, False)
            report.update(vram_used_after_clear_gb=round(used, 2), non_model_gb=round(extra, 2))
    if total - peak < v["headroom_gb"]:
        tiled_peak, _, _ = projected(device, True)
        if total - tiled_peak >= v["headroom_gb"] and set_tiled_decode(prompt):
            report["tiled"] = True
            peak = tiled_peak
    report["projected_peak_gb"] = round(peak, 2)
    report["headroom_gb"] = round(total - peak, 2)
    report["ok"] = total - peak >= v["headroom_gb"]
    if not report["ok"]:
        report["message"] = (f"[MODEL_LOCK] not enough VRAM for {n} frames: projected peak {peak:.1f} GB of "
                             f"{total:.1f} GB leaves {total - peak:.1f} GB (< {v['headroom_gb']} GB) even after "
                             f"clearing non-model memory and tiled decode. Failing fast so another worker takes it")
    return report


# ------------------------------------------------------------------------------------- resident state
class ResidentState:
    """What this worker's ComfyUI holds. One per process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.signature = None        # loader-chain signature of the last successful job
        self.idle_bytes = 0          # VRAM in use after that job (model + context)
        self.loads = 0
        self.reuses = 0

    def check(self, signature):
        """Before queueing: ("reused" | "cold", None) or ("mismatch", message)."""
        with self.lock:
            if self.signature is None:
                return "cold", None
            if signature == self.signature:
                return "reused", None
            changed = sorted(set(signature) ^ set(self.signature) |
                             {k for k in set(signature) & set(self.signature) if signature[k] != self.signature[k]})
            return "mismatch", (f"[MODEL] signature differs from the resident model (nodes {', '.join(changed)}): "
                                "refusing to load a second copy. This is a bug in the locked settings; "
                                "the worker is refreshed")

    def after_job(self, signature, ok, oom, idle_bytes, status):
        with self.lock:
            if oom:
                self.signature, self.idle_bytes = None, 0       # ComfyUI unloaded every model on the OOM
                return
            if ok:
                self.signature = signature
                if idle_bytes:
                    self.idle_bytes = idle_bytes
                if status == "cold":
                    self.loads += 1
                else:
                    self.reuses += 1

    def forget(self):
        with self.lock:
            self.signature, self.idle_bytes = None, 0


STATE = ResidentState()
