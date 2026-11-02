"""v28 job inputs for calmer motion and post-processing. All optional; a job without them is returned unchanged
(same object), so old jobs queue exactly the same workflow as before.

"calm_motion": true             appends DEFAULT_MOTION_PROMPT to the prompt and DEFAULT_MOTION_NEGATIVE to the
                                negative prompt (default false)
"motion_prompt": "..."          replaces DEFAULT_MOTION_PROMPT (also works without calm_motion)
"motion_negative_prompt": "..." replaces DEFAULT_MOTION_NEGATIVE (also works without calm_motion)
"nag_scale" / "nag_tau" / "nag_alpha"   WanVideoApplyNAG strength (handler, v25); checked against object_info ranges
"stabilize": true               FFmpeg stabilization of the output (worker_guard/stabilize.py)
"video_info": true              adds frames / fps / duration of the returned video to the result

The sampler runs at cfg 1.0, so a negative prompt only acts through NAG: the motion negative is appended to the
job's negative_prompt (or, without one, to the workflow's own negative), which turns NAG on in the handler.
Mouth, lip, talking, speaking and blinking terms are never added to the negative (they would fight the lip-sync).
"""
import logging
import re

logger = logging.getLogger("worker_guard")

DEFAULT_MOTION_PROMPT = (
    "the person sits still facing the camera, head steady and centered, calm natural speaking, relaxed face, "
    "subtle gentle expressions, eyes looking at the camera, hands resting still, static locked-off camera, "
    "tripod shot, fixed framing"
)
DEFAULT_MOTION_NEGATIVE = (
    "excessive head movement, head shaking, head bobbing, nodding, swaying, rocking body, turning head, "
    "tilting head, looking away, looking sideways, eyes closed, exaggerated facial expressions, raised eyebrows, "
    "grimacing, hand gestures, pointing, waving, moving arms, leaning forward, camera shake, camera movement, "
    "camera zoom, zoom in, zoom out, camera pan, handheld camera, moving background, jitter, flicker"
)
FORBIDDEN_NEGATIVE = re.compile(r"\b(mouth|mouths|lip|lips|talking|talk|speaking|speak|blink|blinks|blinking)\b",
                                re.IGNORECASE)
# handler.py defaults
HANDLER_DEFAULT_PROMPT = "A person talking naturally"
PROMPT_NODE = "241"
# WanVideoApplyNAG ranges from /object_info of wrapper 088128b (defaults 11.0 / 2.5 / 0.25, set in handler.py)
NAG_RANGES = {"nag_scale": (0.0, 100.0), "nag_tau": (0.0, 10.0), "nag_alpha": (0.0, 1.0)}
BOOL_INPUTS = ("calm_motion", "stabilize", "video_info")
TEXT_INPUTS = ("motion_prompt", "motion_negative_prompt")


def _terms(text):
    return [t.strip() for t in str(text).split(",") if t.strip()]


def check(job_input):
    """Error message for the v28 inputs of this job, or None."""
    for key in BOOL_INPUTS:
        if key in job_input and not isinstance(job_input[key], bool):
            return f"[MOTION] {key} must be true or false (got {job_input[key]!r})"
    for key in TEXT_INPUTS:
        if key in job_input and not isinstance(job_input[key], str):
            return f"[MOTION] {key} must be a string"
    for key, (low, high) in NAG_RANGES.items():
        if key in job_input:
            value = job_input[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                return f"[MOTION] {key} must be a number {low:g}..{high:g} (got {value!r})"
    return None


def negative_terms_to_add(motion_negative, base_negative):
    """Motion negative terms without forbidden ones and without terms the base negative already has."""
    present = {t.lower() for t in _terms(base_negative)}
    added, dropped = [], []
    for term in _terms(motion_negative):
        if FORBIDDEN_NEGATIVE.search(term):
            dropped.append(term)
        elif term.lower() not in present:
            present.add(term.lower())
            added.append(term)
    if dropped:
        logger.warning(f"[MOTION] not added to the negative (mouth / lip / talking / blinking terms): {dropped}")
    return added


def append_to_prompt(prompt, addition):
    """Appends to every '|' segment, so every window of a multi-prompt gets it."""
    return "|".join(f"{seg.rstrip().rstrip(',')}, {addition}" if seg.strip() else seg for seg in prompt.split("|"))


def apply(job_input, base_negative):
    """Returns the job input with the motion prompts applied (same object when there is nothing to do).
    base_negative: the workflow's node 241 negative prompt, used when the job sends none."""
    calm = job_input.get("calm_motion") is True
    positive = job_input.get("motion_prompt") or (DEFAULT_MOTION_PROMPT if calm else None)
    negative = job_input.get("motion_negative_prompt") or (DEFAULT_MOTION_NEGATIVE if calm else None)
    if not positive and not negative:
        return job_input
    out = dict(job_input)
    prompt = str(job_input.get("prompt", HANDLER_DEFAULT_PROMPT))
    added_neg = []
    if positive:
        out["prompt"] = append_to_prompt(prompt, positive)
    if negative:
        if "|" in prompt:
            logger.warning("[MOTION] motion negative skipped: '|' multi-prompts run without NAG (handler)")
        else:
            base = job_input.get("negative_prompt") or base_negative or ""
            added_neg = negative_terms_to_add(negative, base)
            if added_neg:
                out["negative_prompt"] = ", ".join(_terms(base) + added_neg)
    logger.info(f"[MOTION] {'calm_motion on' if calm else 'motion prompts'}: +{len(_terms(positive or ''))} positive "
                f"terms, +{len(added_neg)} negative terms{' (NAG)' if added_neg else ''}")
    return out
