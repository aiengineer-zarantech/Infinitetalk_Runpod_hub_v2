"""Offload settings for video jobs.

Default: offloading OFF (blocks_to_swap 0, force_offload False), except 720p on GPUs below
44 GB (blocks_to_swap 25, force_offload True: 720p without offloading ran out of memory on 32 GB;
where it failed is not measured yet, see TESTING.md step 2).
Hard rule: 480p on GPUs >= FULL_GPU_MIN_VRAM_GB (28) always runs with 0 / False, whatever the
client, manual mode or OFFLOAD_TIERS say. Only the emergency OOM retry may change it.
The client's values are ignored. GPUs below MIN_VRAM_GB (default 28) are rejected (per job here,
at startup in gpu_check.py). On an OutOfMemoryError the job is retried once with an emergency
swap: 0 blocks -> OOM_EMERGENCY_BLOCKS (default 6), otherwise 10 more blocks (max 40).
"offload_mode": "manual" (client values as sent) is honoured only with OFFLOAD_ALLOW_MANUAL=1.
The handler turns blocks_to_swap / force_offload into the workflow settings exactly as before;
enforce_final() checks the workflow right before /prompt and fixes it if it does not match.
Tunable without a rebuild: OFFLOAD_TIERS (JSON), OFFLOAD_480P_MAX_PIXELS, MIN_VRAM_GB.
"""
import copy
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("worker_guard")

MAX_BLOCKS = 40
DEFAULT_MIN_VRAM_GB = 28
DEFAULT_OOM_EMERGENCY_BLOCKS = 6
# OutOfMemoryError in these nodes is retried once with the emergency swap
OOM_RETRY_NODES = frozenset({"WanVideoSampler", "WanVideoDecode"})
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
class Decision:
    mode: str
    blocks_to_swap: int
    force_offload: bool
    vram_gb: float
    width: int
    height: int
    row: str = None


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


def emergency_blocks():
    return min(MAX_BLOCKS, max(0, _env_number("OOM_EMERGENCY_BLOCKS", DEFAULT_OOM_EMERGENCY_BLOCKS, int)))


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


def decide(job_input, vram_total_bytes, gpu_name="GPU"):
    """Returns (Decision, None), or (None, error message) when this GPU must not run the job."""
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
            return Decision(
                mode="auto", blocks_to_swap=settings["blocks_to_swap"], force_offload=settings["force_offload"],
                vram_gb=vram_gb, width=width, height=height, row=row,
            ), None
    return None, (
        f"[OFFLOAD] {gpu_name} has {vram_gb:.1f} GB VRAM; auto offload needs at least "
        f"{tiers[-1]['min_vram_gb']:.0f} GB. Job not run."
    )


def apply(job_input, decision):
    """Returns the job input the handler should see."""
    if decision.mode != "auto":
        return job_input
    out = dict(job_input)
    out["blocks_to_swap"] = decision.blocks_to_swap
    out["force_offload"] = decision.force_offload
    # Let the handler derive load_device from blocks_to_swap (offload_device when > 0)
    out.pop("load_device", None)
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


EMERGENCY_BLOCK_STEP = 10


def emergency_settings(blocks_to_swap, force_offload=False):
    """(blocks_to_swap, force_offload) for the one emergency retry after an OOM.
    No offloading -> OOM_EMERGENCY_BLOCKS (6); already swapping -> 10 more blocks (max 40).
    Always swaps more than the attempt that ran out of memory (unless already at 40);
    force_offload stays as it was for that attempt."""
    if blocks_to_swap < emergency_blocks():
        return emergency_blocks(), force_offload
    return min(blocks_to_swap + EMERGENCY_BLOCK_STEP, MAX_BLOCKS), force_offload
