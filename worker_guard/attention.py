"""Attention mode of the InfiniteTalk model loader (WanVideoModelLoader, node 122).

ATTENTION_MODE: "sageattn" (default, what the workflows already contain) or "sdpa".
The installed sageattention 2.2.0 has kernels for Blackwell (sm_120a) only, so on an A40 (sm_86) or an
L40 / L40S / RTX 6000 Ada (sm_89) its first call fails. gpu_check.py therefore runs one small sageattn call
on cuda:0 at startup and compares it with torch's SDPA (self_test); the result goes into the GPU info file.
When it is not "ok", every job uses sdpa instead, so no job fails because of the attention kernel.
Node 122 is only changed when the effective mode differs from the workflow's: default jobs on a GPU that
passed the self-test (or without a self-test result) queue exactly the same prompt as before.
"""
import logging
import os

logger = logging.getLogger("worker_guard")

LOADER_CLASS = "WanVideoModelLoader"
DEFAULT_MODE = "sageattn"
FALLBACK_MODE = "sdpa"
MODES = (DEFAULT_MODE, FALLBACK_MODE)
SELF_TEST_OK = "ok"
# sageattn quantizes q/k to int8 and v to fp8: close to SDPA, not equal
SELF_TEST_MIN_COSINE = 0.99
SELF_TEST_SHAPE = (1, 8, 512, 128)  # batch, heads, tokens, head dim (Wan 14B uses head dim 128)


def _valid(value):
    value = str(value).strip().lower()
    return value if value in MODES else None


def requested_mode(job_input, benchmark):
    """Returns (mode, source). Per-job "attention_mode" is honoured only for benchmark jobs."""
    if benchmark and "attention_mode" in job_input:
        mode = _valid(job_input["attention_mode"])
        if mode:
            return mode, "job input"
        logger.warning(f"[ACCEL] job attention_mode={job_input['attention_mode']!r} ignored (allowed: "
                       f"{', '.join(MODES)})")
    raw = os.environ.get("ATTENTION_MODE")
    if raw is None or raw.strip() == "":
        return DEFAULT_MODE, "default"
    mode = _valid(raw)
    if mode:
        return mode, "ATTENTION_MODE"
    logger.warning(f"[ACCEL] invalid ATTENTION_MODE={raw!r} (allowed: {', '.join(MODES)}); using {DEFAULT_MODE}")
    return DEFAULT_MODE, "default"


def decide(job_input, benchmark, gpu_info):
    """Returns (mode, note) for this job. gpu_info: the GPU info file written by gpu_check.py, or None."""
    mode, source = requested_mode(job_input, benchmark)
    status = (gpu_info or {}).get("sageattn")
    if mode == DEFAULT_MODE and status is not None and status != SELF_TEST_OK:
        gpu = (gpu_info or {}).get("name") or "this GPU"
        return FALLBACK_MODE, f"{FALLBACK_MODE} (sageattn unavailable on {gpu}: {status})"
    if mode == DEFAULT_MODE:
        test = "self-test ok" if status == SELF_TEST_OK else "no self-test result"
        return mode, f"{mode} ({source}, {test})"
    return mode, f"{mode} ({source})"


def apply(prompt, mode):
    """Sets attention_mode on every WanVideoModelLoader whose value differs. Returns the changed node ids."""
    changed = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or node.get("class_type") != LOADER_CLASS:
            continue
        inputs = node.setdefault("inputs", {})
        if inputs.get("attention_mode") != mode:
            inputs["attention_mode"] = mode
            changed.append(str(node_id))
    return changed


def self_test(torch):
    """One sageattn call on cuda:0 against torch SDPA. Returns "ok" or "failed: <reason>". Never raises."""
    try:
        from sageattention import sageattn

        gen = torch.Generator(device="cuda:0").manual_seed(0)
        q, k, v = (torch.randn(SELF_TEST_SHAPE, generator=gen, device="cuda:0", dtype=torch.bfloat16)
                   for _ in range(3))
        out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize(0)
        if tuple(out.shape) != tuple(ref.shape):
            return f"failed: output shape {tuple(out.shape)}, expected {tuple(ref.shape)}"
        if not bool(torch.isfinite(out).all().item()):
            return "failed: non-finite output"
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        if cos < SELF_TEST_MIN_COSINE:
            return f"failed: cosine similarity to sdpa {cos:.4f} < {SELF_TEST_MIN_COSINE}"
        return SELF_TEST_OK
    except BaseException as e:  # a missing kernel image raises RuntimeError / torch.AcceleratorError
        lines = str(e).strip().splitlines()
        return f"failed: {type(e).__name__}: {lines[0][:200] if lines else ''}"


def package_versions():
    """Versions for the [ACCEL] line (read in the GPU check process, which has torch loaded)."""
    from importlib import metadata

    out = {}
    for name in ("sageattention", "triton"):
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = "not installed"
    return out
