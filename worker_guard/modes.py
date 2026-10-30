"""Job modes added in v24/v26. A job without "mode" (and without "upscale") is returned unchanged
(same object), so old jobs queue exactly the same workflow as before.

"mode": "fast480"       I2V single person only. Size 848x480, or 480x848 with "aspect": "9:16".
                        The reference image is center-cropped to that size by the workflow itself
                        (node 281 ImageResizeKJv2: keep_proportion "crop", crop_position "center"), never stretched.
                        Any width/height the client sends is replaced. Offloading follows the 480p rule.
                        v26: then upscaled to 1280x720 / 720x1280 with SeedVR2 ("upscale": "720p", the default).
"mode": "upscale_only"  v26: no InfiniteTalk run; upscales the video in "video_base64", "video_url" or
                        "video_path" (worker_guard/upscale.py).

"upscale": "720p" | "none"     only with fast480 / upscale_only; "none" keeps the 480p output (fast480 only)
"upscale_model": "3b" | "7b"   SeedVR2 size (default "3b")
"""
import logging

logger = logging.getLogger("worker_guard")

STANDARD = "standard"
FAST480 = "fast480"
UPSCALE_ONLY = "upscale_only"
FAST480_SIZES = {"16:9": (848, 480), "9:16": (480, 848)}

UPSCALE_720P = "720p"
UPSCALE_NONE = "none"
UPSCALE_MODELS = ("3b", "7b")
DEFAULT_UPSCALE_MODEL = "3b"
UPSCALE_VIDEO_INPUTS = ("video_base64", "video_url", "video_path")


def _mode(job_input):
    return str((job_input or {}).get("mode", "")).strip().lower()


def job_mode(job_input):
    mode = _mode(job_input)
    return mode if mode in (FAST480, UPSCALE_ONLY) else STANDARD


def upscale_settings(job_input):
    """{"model": "3b" | "7b"} when this (prepared) job is upscaled, else None."""
    mode = job_mode(job_input)
    if mode not in (FAST480, UPSCALE_ONLY):
        return None
    if str(job_input.get("upscale", UPSCALE_720P)).strip().lower() != UPSCALE_720P:
        return None
    return {"model": str(job_input.get("upscale_model", DEFAULT_UPSCALE_MODEL)).strip().lower()}


def _check_upscale(job_input, mode):
    """Error message for the upscale inputs of this job, or None."""
    upscale = str(job_input.get("upscale", UPSCALE_720P)).strip().lower()
    if upscale not in (UPSCALE_720P, UPSCALE_NONE):
        return f"[MODE] upscale must be '{UPSCALE_720P}' or '{UPSCALE_NONE}' (got {job_input.get('upscale')!r})"
    if mode == UPSCALE_ONLY and upscale == UPSCALE_NONE:
        return "[MODE] upscale_only with upscale 'none' has nothing to do"
    model = str(job_input.get("upscale_model", DEFAULT_UPSCALE_MODEL)).strip().lower()
    if model not in UPSCALE_MODELS:
        return f"[MODE] upscale_model must be one of {list(UPSCALE_MODELS)} (got {job_input.get('upscale_model')!r})"
    return None


def prepare_job_input(job_input):
    """Returns (job_input, None) or (job_input, error message). Idempotent."""
    if "mode" not in job_input:
        if "upscale" in job_input and str(job_input["upscale"]).strip().lower() != UPSCALE_NONE:
            return job_input, "[MODE] upscale is only available with mode 'fast480' or 'upscale_only'"
        return job_input, None
    mode = _mode(job_input)
    if mode == STANDARD:
        if "upscale" in job_input and str(job_input["upscale"]).strip().lower() != UPSCALE_NONE:
            return job_input, "[MODE] upscale is only available with mode 'fast480' or 'upscale_only'"
        return job_input, None
    if mode == UPSCALE_ONLY:
        if not any(job_input.get(key) for key in UPSCALE_VIDEO_INPUTS):
            return job_input, f"[MODE] upscale_only needs one of {list(UPSCALE_VIDEO_INPUTS)}"
        return job_input, _check_upscale(job_input, mode)
    if mode != FAST480:
        return job_input, (f"[MODE] unknown mode {job_input.get('mode')!r}; supported: 'fast480', 'upscale_only' "
                           f"(or leave mode out)")
    input_type = job_input.get("input_type", "image")
    person_count = job_input.get("person_count", "single")
    if input_type != "image" or person_count != "single":
        return job_input, (f"[MODE] fast480 supports input_type 'image' with person_count 'single' only "
                           f"(got {input_type!r}, {person_count!r})")
    aspect = str(job_input.get("aspect", "16:9")).strip()
    if aspect not in FAST480_SIZES:
        return job_input, f"[MODE] fast480 aspect must be one of {sorted(FAST480_SIZES)} (got {aspect!r})"
    error = _check_upscale(job_input, mode)
    if error:
        return job_input, error
    width, height = FAST480_SIZES[aspect]
    sent = (job_input.get("width"), job_input.get("height"))
    if sent != (None, None) and sent != (width, height):
        logger.info(f"[MODE] fast480: width/height {sent[0]}x{sent[1]} from the client replaced by {width}x{height}")
    out = dict(job_input)
    out["width"], out["height"] = width, height
    return out, None
