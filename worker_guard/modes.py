"""Job modes added in v24. A job without "mode" is returned unchanged (same object), so old jobs
queue exactly the same workflow as before.

"mode": "fast480"  I2V single person only. Size 848x480, or 480x848 with "aspect": "9:16".
                   The reference image is center-cropped to that size by the workflow itself
                   (node 281 ImageResizeKJv2: keep_proportion "crop", crop_position "center"), never stretched.
                   Any width/height the client sends is replaced. Offloading follows the 480p rule.
"""
import logging

logger = logging.getLogger("worker_guard")

STANDARD = "standard"
FAST480 = "fast480"
FAST480_SIZES = {"16:9": (848, 480), "9:16": (480, 848)}


def job_mode(job_input):
    return FAST480 if str((job_input or {}).get("mode", "")).strip().lower() == FAST480 else STANDARD


def prepare_job_input(job_input):
    """Returns (job_input, None) or (job_input, error message). Idempotent."""
    if "mode" not in job_input:
        return job_input, None
    mode = str(job_input.get("mode")).strip().lower()
    if mode == STANDARD:
        return job_input, None
    if mode != FAST480:
        return job_input, f"[MODE] unknown mode {job_input.get('mode')!r}; supported: 'fast480' (or leave mode out)"
    input_type = job_input.get("input_type", "image")
    person_count = job_input.get("person_count", "single")
    if input_type != "image" or person_count != "single":
        return job_input, (f"[MODE] fast480 supports input_type 'image' with person_count 'single' only "
                           f"(got {input_type!r}, {person_count!r})")
    aspect = str(job_input.get("aspect", "16:9")).strip()
    if aspect not in FAST480_SIZES:
        return job_input, f"[MODE] fast480 aspect must be one of {sorted(FAST480_SIZES)} (got {aspect!r})"
    width, height = FAST480_SIZES[aspect]
    sent = (job_input.get("width"), job_input.get("height"))
    if sent != (None, None) and sent != (width, height):
        logger.info(f"[MODE] fast480: width/height {sent[0]}x{sent[1]} from the client replaced by {width}x{height}")
    out = dict(job_input)
    out["width"], out["height"] = width, height
    return out, None
