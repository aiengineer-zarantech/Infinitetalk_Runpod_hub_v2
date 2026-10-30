"""Offload settings for video jobs.

Default: offloading OFF (blocks_to_swap 0, force_offload False), except 720p on GPUs below
44 GB (blocks_to_swap 25, force_offload True: 720p without offloading ran out of memory on 32 GB;
where it failed is not measured yet, see TESTING.md step 2).
Hard rule: 480p on GPUs >= FULL_GPU_MIN_VRAM_GB (28) always runs with 0 / False, whatever the
client, manual mode or OFFLOAD_TIERS say. Only the emergency OOM retry may change it.
The client's values are ignored. GPUs below MIN_VRAM_GB (default 28) are rejected (per job here,
at startup in gpu_check.py).
Adaptive (v24): with a calibration (OFFLOAD_CALIBRATION, or DEFAULT_CALIBRATION once measured), jobs
that are not 480p-on-the-GPU get the SMALLEST blocks_to_swap whose predicted peak fits in
OFFLOAD_TARGET_FRACTION (0.85) of the VRAM. Without a calibration the tier values are used.
OOM ladder (v24): on an OutOfMemoryError the job is retried with blocks_to_swap +10, then 25, then 40,
and only after 40 also ran out of memory, once more with VAE tiling on (output may differ).
"offload_mode": "manual" (client values as sent) is honoured only with OFFLOAD_ALLOW_MANUAL=1.
The handler turns blocks_to_swap / force_offload into the workflow settings exactly as before;
enforce_final() checks the workflow right before /prompt and fixes it if it does not match.
Tunable without a rebuild: OFFLOAD_TIERS (JSON), OFFLOAD_CALIBRATION (JSON), OFFLOAD_TARGET_FRACTION,
OFFLOAD_480P_MAX_PIXELS, MIN_VRAM_GB.
BLOCKS_TO_SWAP (v27): unset = the auto logic above. A value 0..40 replaces the tier / adaptive choice for auto jobs
(force_offload False, OOM ladder still on), except 480p on >= 28 GB, which stays 0.
"""
import copy
import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("worker_guard")

MAX_BLOCKS = 40
DEFAULT_MIN_VRAM_GB = 28
# OutOfMemoryError in these nodes starts the OOM ladder
OOM_RETRY_NODES = frozenset({"WanVideoSampler", "WanVideoDecode"})
LADDER_STEP = 10
LADDER_FIXED = (25, MAX_BLOCKS)
DEFAULT_TARGET_FRACTION = 0.85
# Until the A.4 runs measure it (wrapper log "Transformer blocks on cpu: X MB" / blocks_to_swap): a Wan 14B
# block in fp8 is ~0.38 GB, the InfiniteTalk audio layers add a little. Used for the RAM estimate only.
BLOCK_GB_ESTIMATE = 0.44
DEFAULT_WINDOW_FRAMES = 81
# Filled in from the TESTING.md Phase A measurements; None = not calibrated (tier values are used)
DEFAULT_CALIBRATION = None
STAGES = ("sampler", "vae")
DEFAULT_480P_MAX_PIXELS = 900 * 500
# handler.py defaults when the job sends no width/height
HANDLER_DEFAULT_SIZE = 512

# Offloading off: only GPUs with >= 32 GB are used (they report ~31.4 GB); 480p peaks at ~24-26 GB
DEFAULT_TIERS = [
    {"min_vram_gb": 44,
     "480p": {"blocks_to_swap": 0, "force_offload": False},
     "720p": {"blocks_to_swap": 0, "force_offload": False}},
    # 720p without offloading ran out of memory on a 32 GB RTX 5090 (owner report)
    {"min_vram_gb": DEFAULT_MIN_VRAM_GB,
     "480p": {"blocks_to_swap": 0, "force_offload": False},
     "720p": {"blocks_to_swap": 25, "force_offload": True}},
]
ROWS = ("480p", "720p")
# 480p on these GPUs always runs fully on the GPU (854x480 peaks at ~24-26 GB). Not env-tunable.
FULL_GPU_MIN_VRAM_GB = 28
FULL_GPU = {"blocks_to_swap": 0, "force_offload": False}
# Prompt nodes the handler fills in: width, height
WIDTH_NODE, HEIGHT_NODE = "245", "246"


@dataclass
class Prediction:
    calibrated: bool
    budget_gb: float
    foreign_gb: float
    predicted_gb: float = None
    fits: bool = True
    note: str = ""


@dataclass
class Decision:
    mode: str
    blocks_to_swap: int
    force_offload: bool
    vram_gb: float
    width: int
    height: int
    row: str = None
    prediction: Prediction = None
    vae_tiling: bool = False
    tried: list = field(default_factory=list)


def validate_tiers(tiers):
    """Returns the tiers sorted by min_vram_gb (highest first), or raises ValueError."""
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("must be a non-empty JSON list")
    out = []
    for i, tier in enumerate(tiers):
        if not isinstance(tier, dict):
            raise ValueError(f"tier {i} is not an object")
        min_gb = tier.get("min_vram_gb")
        if isinstance(min_gb, bool) or not isinstance(min_gb, (int, float)) or min_gb < 0:
            raise ValueError(f"tier {i}: min_vram_gb must be a number >= 0")
        clean = {"min_vram_gb": float(min_gb)}
        for row in ROWS:
            settings = tier.get(row)
            if not isinstance(settings, dict):
                raise ValueError(f"tier {i}: missing {row!r} object")
            blocks = settings.get("blocks_to_swap")
            force = settings.get("force_offload")
            if isinstance(blocks, bool) or not isinstance(blocks, int) or not 0 <= blocks <= MAX_BLOCKS:
                raise ValueError(f"tier {i} {row}: blocks_to_swap must be an integer 0..{MAX_BLOCKS}")
            if not isinstance(force, bool):
                raise ValueError(f"tier {i} {row}: force_offload must be true or false")
            clean[row] = {"blocks_to_swap": blocks, "force_offload": force}
        out.append(clean)
    return sorted(out, key=lambda t: t["min_vram_gb"], reverse=True)


def load_tiers():
    raw = os.environ.get("OFFLOAD_TIERS")
    if raw and raw.strip():
        try:
            return validate_tiers(json.loads(raw))
        except (ValueError, TypeError) as e:
            logger.error(f"[OFFLOAD] invalid OFFLOAD_TIERS ({e}); using built-in defaults")
    return validate_tiers(copy.deepcopy(DEFAULT_TIERS))


def max_480p_pixels():
    raw = os.environ.get("OFFLOAD_480P_MAX_PIXELS")
    if raw and raw.strip():
        try:
            return int(raw)
        except ValueError:
            logger.error(f"[OFFLOAD] invalid OFFLOAD_480P_MAX_PIXELS={raw!r}; using {DEFAULT_480P_MAX_PIXELS}")
    return DEFAULT_480P_MAX_PIXELS


def resolution_row(width, height):
    return "480p" if width * height <= max_480p_pixels() else "720p"


def full_gpu_required(vram_gb, row):
    """True if this job must run with blocks_to_swap 0 / force_offload False (outside the OOM retry)."""
    return row == "480p" and vram_gb >= FULL_GPU_MIN_VRAM_GB


def _env_number(name, default, cast):
    raw = os.environ.get(name)
    if raw and raw.strip():
        try:
            return cast(raw)
        except ValueError:
            logger.error(f"[OFFLOAD] invalid {name}={raw!r}; using {default}")
    return default


def min_vram_gb():
    return _env_number("MIN_VRAM_GB", float(DEFAULT_MIN_VRAM_GB), float)


def blocks_override():
    """BLOCKS_TO_SWAP as an int 0..MAX_BLOCKS, or None when unset or invalid (invalid is logged)."""
    raw = os.environ.get("BLOCKS_TO_SWAP")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if not 0 <= value <= MAX_BLOCKS:
        logger.error(f"[OFFLOAD] invalid BLOCKS_TO_SWAP={raw!r} (0..{MAX_BLOCKS}); ignored, auto logic used")
        return None
    return value


def target_fraction():
    value = _env_number("OFFLOAD_TARGET_FRACTION", DEFAULT_TARGET_FRACTION, float)
    if not 0.5 <= value <= 0.95:
        logger.error(f"[OFFLOAD] OFFLOAD_TARGET_FRACTION={value} outside 0.5..0.95; using {DEFAULT_TARGET_FRACTION}")
        return DEFAULT_TARGET_FRACTION
    return value


def validate_calibration(cal):
    """Returns the calibration as floats, or raises ValueError.
    {"block_gb": B, "sampler": {"base_gb": a, "gb_per_ktoken": b}, "vae": {"base_gb": c, "gb_per_ktoken": d}}
    Stage peak with N blocks swapped = base_gb + gb_per_ktoken * latent tokens per window / 1000 - N * block_gb."""
    def number(value, what, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (positive and value <= 0):
            raise ValueError(f"{what} must be a number {'> 0' if positive else '>= 0'}")
        return float(value)

    if not isinstance(cal, dict):
        raise ValueError("must be a JSON object")
    out = {"block_gb": number(cal.get("block_gb"), "block_gb", positive=True)}
    for stage in STAGES:
        settings = cal.get(stage)
        if not isinstance(settings, dict):
            raise ValueError(f"missing {stage!r} object")
        out[stage] = {k: number(settings.get(k), f"{stage}.{k}") for k in ("base_gb", "gb_per_ktoken")}
    return out


def load_calibration():
    raw = os.environ.get("OFFLOAD_CALIBRATION")
    if raw and raw.strip():
        try:
            return validate_calibration(json.loads(raw))
        except (ValueError, TypeError) as e:
            logger.error(f"[OFFLOAD] invalid OFFLOAD_CALIBRATION ({e}); not calibrated, using the tier values")
            return None
    return validate_calibration(copy.deepcopy(DEFAULT_CALIBRATION)) if DEFAULT_CALIBRATION else None


def block_gb():
    """(GB per swapped block, 'calibrated' or 'estimate')."""
    cal = load_calibration()
    return (cal["block_gb"], "calibrated") if cal else (BLOCK_GB_ESTIMATE, "estimate")


def latent_tokens(width, height, window_frames=DEFAULT_WINDOW_FRAMES):
    """Transformer tokens per InfiniteTalk window: (W/16) x (H/16) x latent frames (81 frames -> 21)."""
    return (width // 16) * (height // 16) * ((window_frames - 1) // 4 + 1)


def predicted_peak_gb(cal, width, height, blocks, window_frames=DEFAULT_WINDOW_FRAMES):
    ktokens = latent_tokens(width, height, window_frames) / 1000
    return max(cal[s]["base_gb"] + cal[s]["gb_per_ktoken"] * ktokens for s in STAGES) - blocks * cal["block_gb"]


def adaptive_blocks(cal, width, height, budget_gb, window_frames=DEFAULT_WINDOW_FRAMES):
    """(smallest blocks_to_swap whose predicted peak fits the budget, predicted GB, fits). 40 if none fits."""
    for blocks in range(MAX_BLOCKS + 1):
        peak = predicted_peak_gb(cal, width, height, blocks, window_frames)
        if peak <= budget_gb:
            return blocks, peak, True
    return MAX_BLOCKS, predicted_peak_gb(cal, width, height, MAX_BLOCKS, window_frames), False


def manual_allowed():
    return os.environ.get("OFFLOAD_ALLOW_MANUAL", "").strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mode_of(job_input):
    return "manual" if str(job_input.get("offload_mode", "auto")).strip().lower() == "manual" else "auto"


def vram_reject(vram_total_bytes, gpu_name="GPU"):
    """Error message if this GPU is below MIN_VRAM_GB (no job may run on it), else None."""
    vram_gb = vram_total_bytes / (1024 ** 3)
    minimum = min_vram_gb()
    if vram_gb < minimum:
        return f"[OFFLOAD] {gpu_name} has {vram_gb:.1f} GB VRAM; minimum is {minimum:.0f} GB. Job not run."
    return None


def decide(job_input, vram_total_bytes, gpu_name="GPU", vram_used_bytes=None, resident_bytes=0):
    """Returns (Decision, None), or (None, error message) when this GPU must not run the job.
    vram_used_bytes: VRAM in use at job start; resident_bytes: VRAM our own models held at the end of the
    previous job on this worker. The difference (if positive) is not ours and is taken off the budget."""
    vram_gb = vram_total_bytes / (1024 ** 3)
    width = _as_int(job_input.get("width", HANDLER_DEFAULT_SIZE), HANDLER_DEFAULT_SIZE)
    height = _as_int(job_input.get("height", HANDLER_DEFAULT_SIZE), HANDLER_DEFAULT_SIZE)
    reject = vram_reject(vram_total_bytes, gpu_name)
    if reject:
        return None, reject
    row = resolution_row(width, height)
    full_gpu = full_gpu_required(vram_gb, row)
    manual = mode_of(job_input) == "manual"
    if manual and not manual_allowed():
        logger.warning("[OFFLOAD] manual mode ignored (OFFLOAD_ALLOW_MANUAL=0); client offload values are not used")
        manual = False
    elif manual and full_gpu:
        logger.warning(f"[OFFLOAD] manual values ignored: {row} on {vram_gb:.1f} GB always runs with "
                       "blocks_to_swap 0, force_offload False")
        manual = False
    if manual:
        # Reported only; the handler reads the client's values itself (defaults 0 / True)
        return Decision(
            mode="manual",
            blocks_to_swap=_as_int(job_input.get("blocks_to_swap", 0), 0),
            force_offload=job_input.get("force_offload", True),
            vram_gb=vram_gb, width=width, height=height,
        ), None

    tiers = load_tiers()
    for tier in tiers:
        if vram_gb >= tier["min_vram_gb"]:
            settings = tier[row]
            if full_gpu and settings != FULL_GPU:
                logger.warning(f"[OFFLOAD] OFFLOAD_TIERS {row} row {settings} ignored: {row} on {vram_gb:.1f} GB "
                               "always runs with blocks_to_swap 0, force_offload False")
                settings = FULL_GPU
            decision = Decision(
                mode="auto", blocks_to_swap=settings["blocks_to_swap"], force_offload=settings["force_offload"],
                vram_gb=vram_gb, width=width, height=height, row=row,
            )
            _adapt(decision, vram_used_bytes, resident_bytes, full_gpu)
            _apply_blocks_override(decision, full_gpu)
            return decision, None
    return None, (
        f"[OFFLOAD] {gpu_name} has {vram_gb:.1f} GB VRAM; auto offload needs at least "
        f"{tiers[-1]['min_vram_gb']:.0f} GB. Job not run."
    )


def _adapt(decision, vram_used_bytes, resident_bytes, full_gpu):
    """Sets decision.prediction and, when calibrated (and not a 480p-on-the-GPU job), the adaptive blocks_to_swap."""
    gb = 1024 ** 3
    budget = target_fraction() * decision.vram_gb
    foreign = max(0.0, ((vram_used_bytes or 0) - (resident_bytes or 0)) / gb) if vram_used_bytes is not None else 0.0
    budget -= foreign
    cal = load_calibration()
    pred = Prediction(calibrated=cal is not None, budget_gb=budget, foreign_gb=foreign)
    decision.prediction = pred
    if cal is None:
        pred.note = "not calibrated, tier value used"
        return
    pred.predicted_gb = predicted_peak_gb(cal, decision.width, decision.height, decision.blocks_to_swap)
    if full_gpu:
        pred.fits = pred.predicted_gb <= budget
        pred.note = "480p rule: always 0"
        return
    blocks, pred.predicted_gb, pred.fits = adaptive_blocks(cal, decision.width, decision.height, budget)
    pred.note = "smallest that fits" if pred.fits else f"nothing fits; using {MAX_BLOCKS}"
    # Adaptive jobs keep the model on the GPU after sampling (resident between jobs); nothing after the
    # sampler needs the VRAM (VAE decode runs inside it, video combine runs on the CPU)
    decision.blocks_to_swap, decision.force_offload = blocks, False


def _apply_blocks_override(decision, full_gpu):
    override = blocks_override()
    if override is None:
        return
    size = f"{decision.width}x{decision.height}"
    if full_gpu:
        if override != 0:
            logger.warning(f"[OFFLOAD] BLOCKS_TO_SWAP={override} ignored: {size} on {decision.vram_gb:.1f} GB always runs "
                           "with blocks_to_swap 0, force_offload False")
        return
    logger.info(f"[OFFLOAD] BLOCKS_TO_SWAP={override}: {size} uses blocks_to_swap {override} instead of "
                f"{decision.blocks_to_swap}, force_offload False")
    decision.blocks_to_swap, decision.force_offload = override, False
    pred = decision.prediction
    if pred is not None:
        cal = load_calibration()
        if cal is not None:
            pred.predicted_gb = predicted_peak_gb(cal, decision.width, decision.height, override)
            pred.fits = pred.predicted_gb <= pred.budget_gb
        pred.note = f"BLOCKS_TO_SWAP={override} (env)"


def adaptive_line(decision, actual_peak_bytes, vram_total_bytes):
    """[OFFLOAD] adaptive: predicted peak X GB (Y%), chosen blocks_to_swap N, actual peak Z GB (W%)"""
    total_gb = vram_total_bytes / (1024 ** 3) if vram_total_bytes else decision.vram_gb
    pred = decision.prediction
    if pred is None or pred.predicted_gb is None:
        predicted = "n/a (" + (pred.note if pred else f"{decision.mode} mode") + ")"
    else:
        predicted = f"{pred.predicted_gb:.1f} GB ({100 * pred.predicted_gb / total_gb:.0f}%)"
        if pred.note:
            predicted += f" [{pred.note}]"
    if actual_peak_bytes:
        actual_gb = actual_peak_bytes / (1024 ** 3)
        actual = f"{actual_gb:.1f} GB ({100 * actual_gb / total_gb:.0f}%)"
    else:
        actual = "n/a"
    extra = f", budget {pred.budget_gb:.1f} GB" if pred else ""
    if pred and pred.foreign_gb >= 0.1:
        extra += f" (minus {pred.foreign_gb:.1f} GB not ours at job start)"
    return (f"[OFFLOAD] adaptive: predicted peak {predicted}, chosen blocks_to_swap {decision.blocks_to_swap}, "
            f"actual peak {actual}{extra}")


def apply(job_input, decision):
    """Returns the job input the handler should see."""
    if decision.mode != "auto":
        return job_input
    out = dict(job_input)
    out["blocks_to_swap"] = decision.blocks_to_swap
    out["force_offload"] = decision.force_offload
    # Let the handler derive load_device from blocks_to_swap (offload_device when > 0)
    out.pop("load_device", None)
    if decision.vae_tiling:
        # Last OOM ladder step only (never a normal job): the handler's VAE tiling input
        out["vae_tiling"] = True
    return out


def _nodes(prompt, class_type):
    return [(node_id, node.setdefault("inputs", {})) for node_id, node in prompt.items()
            if isinstance(node, dict) and node.get("class_type") == class_type]


def _final_values(prompt):
    swap = _nodes(prompt, "WanVideoBlockSwap")
    sampler = _nodes(prompt, "WanVideoSampler")
    loader = _nodes(prompt, "WanVideoModelLoader")
    blocks = _as_int(swap[0][1].get("blocks_to_swap", 0), 0) if swap else 0
    force = sampler[0][1].get("force_offload") if sampler else None
    load_device = loader[0][1].get("load_device") if loader else None
    return blocks, force, load_device


def _set_offload(prompt, blocks_to_swap, force_offload):
    """Same workflow settings the handler writes for these values."""
    offload = blocks_to_swap > 0
    for _, inputs in _nodes(prompt, "WanVideoBlockSwap"):
        inputs["blocks_to_swap"] = blocks_to_swap
        inputs["offload_img_emb"] = offload
        inputs["offload_txt_emb"] = offload
    for _, inputs in _nodes(prompt, "WanVideoSampler"):
        inputs["force_offload"] = force_offload
    for _, inputs in _nodes(prompt, "WanVideoModelLoader"):
        inputs["load_device"] = "offload_device" if offload else "main_device"


def _node_int(prompt, node_id, default):
    return _as_int(((prompt.get(node_id) or {}).get("inputs") or {}).get("value"), default)


def enforce_final(prompt, vram_gb, expected=None, oom_retry=False):
    """Last check before the workflow goes to ComfyUI /prompt. Fixes the prompt in place (and logs an
    error) when it does not match `expected` (blocks_to_swap, force_offload), or when a 480p job on a
    >= 28 GB GPU would offload outside the OOM retry. vram_gb None (unknown) counts as >= 28 GB, since
    smaller GPUs are rejected. Logs the final values and returns (blocks_to_swap, force_offload)."""
    width = _node_int(prompt, WIDTH_NODE, HANDLER_DEFAULT_SIZE)
    height = _node_int(prompt, HEIGHT_NODE, HANDLER_DEFAULT_SIZE)
    shown_vram = "unknown" if vram_gb is None else f"{vram_gb:.1f} GB"
    full_gpu = not oom_retry and full_gpu_required(FULL_GPU_MIN_VRAM_GB if vram_gb is None else vram_gb,
                                                   resolution_row(width, height))
    blocks, force, load_device = _final_values(prompt)
    if full_gpu:
        expected = (0, False)
    wrong = expected is not None and ((blocks, force) != tuple(expected) or
                                      (full_gpu and load_device not in (None, "main_device")))
    if wrong:
        logger.error(
            f"[OFFLOAD] final workflow has blocks_to_swap {blocks}, force_offload {force}, load_device {load_device} "
            f"(GPU {shown_vram}, {width}x{height}, oom_retry {'yes' if oom_retry else 'no'}); expected "
            f"blocks_to_swap {expected[0]}, force_offload {expected[1]}. Fixed before queueing"
        )
        _set_offload(prompt, expected[0], expected[1])
        blocks, force, load_device = _final_values(prompt)
    swap = _nodes(prompt, "WanVideoBlockSwap")
    node_id, inputs = swap[0] if swap else ("134", {})
    shown = {k: inputs.get(k) for k in ("blocks_to_swap", "offload_img_emb", "offload_txt_emb")}
    logger.info(f"[OFFLOAD] final: blocks_to_swap {blocks}, force_offload {force}, node {node_id} {shown}, "
                f"load_device {load_device}")
    return blocks, force


def is_oom(result):
    """True if the handler result is a ComfyUI OutOfMemoryError in a retryable node."""
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    return "OutOfMemoryError" in str(details.get("exception_type") or "") and details.get("node_type") in OOM_RETRY_NODES


def oom_ladder(blocks_to_swap):
    """Retry steps after an OOM at blocks_to_swap: [(blocks, vae_tiling), ...].
    +10, then 25, then 40 (each only if more than the step before), then 40 with VAE tiling,
    which therefore only ever runs after a 40-block attempt ran out of memory."""
    steps, previous = [], blocks_to_swap
    for blocks in (blocks_to_swap + LADDER_STEP,) + LADDER_FIXED:
        blocks = min(blocks, MAX_BLOCKS)
        if blocks > previous:
            steps.append((blocks, False))
            previous = blocks
    steps.append((MAX_BLOCKS, True))
    return steps
