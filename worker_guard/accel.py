"""Speed options that do not touch steps, cfg, shift, scheduler, resolution, LoRA or sampler settings.

- torch.compile (TORCH_COMPILE=1, default): the I2V jobs use I2V_*_compiled.json, which are the
  original workflows plus ONE added input, node 122 "compile_args" -> node 177 (the wrapper's own
  WanVideoTorchCompileSettings node, already present but unconnected in the originals).
  The handler still loads the original file; the additions from the compiled file are applied to
  the queued prompt (the handler only changes values of existing inputs, so the result is the
  same as loading the compiled file). TORCH_COMPILE=0 queues the original workflow exactly.
- FP8_FAST=1 (test only, default 0): node 122 quantization fp8_e4m3fn -> fp8_e4m3fn_fast.
  This DOES change numerics, so it is never on by default.
- "benchmark": true trims the audio to the first BENCHMARK_SECONDS seconds.
"""
import base64
import copy
import io
import json
import logging
import os
import tempfile
import wave

logger = logging.getLogger("worker_guard")

LOADER_NODE = "122"
COMPILE_NODE = "177"
SAMPLER_NODE = "128"
COMPILED_WORKFLOWS = {
    "I2V_single.json": "I2V_single_compiled.json",
    "I2V_multi.json": "I2V_multi_compiled.json",
}
FP8_FROM, FP8_TO = "fp8_e4m3fn", "fp8_e4m3fn_fast"
# Wrapper tooltip: "e4m3fn generally can not be torch.compiled on compute capability < 8.9"
MIN_COMPILE_CAPABILITY = (8, 9)
BENCHMARK_SECONDS = 10
BENCHMARK_MAX_FRAME = BENCHMARK_SECONDS * 25 + 81  # handler: int(duration * 25) + 81

# Set after a compile failure: compile stays off for the rest of this worker's life
_compile_disabled_reason = None
_workflow_cache = {}


def env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def workflow_dir():
    return os.environ.get("WORKFLOW_DIR", "/")


def workflow_name(input_type, person_count):
    """Same mapping as handler.get_workflow_path."""
    prefix = "I2V" if input_type == "image" else "V2V"
    return f"{prefix}_{'single' if person_count == 'single' else 'multi'}.json"


def gpu_info():
    path = os.environ.get("WORKER_GPU_INFO", "/tmp/worker_gpu.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _capability(info):
    try:
        major, minor = str(info["capability"]).split(".")[:2]
        return int(major), int(minor)
    except (KeyError, TypeError, ValueError):
        return None


def compile_decision(job_input, benchmark):
    """Returns (on, note). Per-job "torch_compile" is honoured only for benchmark jobs."""
    requested = env_flag("TORCH_COMPILE", True)
    source = "TORCH_COMPILE"
    if benchmark and "torch_compile" in job_input:
        requested = bool(job_input["torch_compile"])
        source = "job input"
    if not requested:
        return False, f"off ({source})"
    if _compile_disabled_reason:
        return False, f"off ({_compile_disabled_reason})"
    cap = _capability(gpu_info() or {})
    if cap is not None and cap < MIN_COMPILE_CAPABILITY:
        return False, f"off (compute capability {cap[0]}.{cap[1]} < 8.9: fp8_e4m3fn cannot be compiled)"
    return True, "on"


def fp8_decision(job_input, benchmark):
    if benchmark and "fp8_fast" in job_input:
        return bool(job_input["fp8_fast"])
    return env_flag("FP8_FAST", False)


def disable_compile(reason):
    global _compile_disabled_reason
    _compile_disabled_reason = reason


def load_workflow(name):
    path = os.path.join(workflow_dir(), name)
    if path not in _workflow_cache:
        with open(path, encoding="utf-8") as f:
            _workflow_cache[path] = json.load(f)
    return copy.deepcopy(_workflow_cache[path])


def workflow_delta(original, variant):
    """Additions that turn `original` into `variant`: (new nodes, {node_id: {input: value}}).
    Raises ValueError if the variant changes or removes anything that exists in the original."""
    new_nodes, new_inputs = {}, {}
    for node_id, node in original.items():
        if node_id not in variant:
            raise ValueError(f"variant removes node {node_id}")
        v_node = variant[node_id]
        for key in set(node) | set(v_node):
            if key != "inputs" and node.get(key) != v_node.get(key):
                raise ValueError(f"variant changes node {node_id} field {key!r}")
        inputs, v_inputs = node.get("inputs", {}), v_node.get("inputs", {})
        for key, value in inputs.items():
            if key not in v_inputs:
                raise ValueError(f"variant removes input {node_id}.{key}")
            if v_inputs[key] != value:
                raise ValueError(f"variant changes input {node_id}.{key}")
        added = {k: v for k, v in v_inputs.items() if k not in inputs}
        if added:
            new_inputs[node_id] = added
    for node_id, node in variant.items():
        if node_id not in original:
            new_nodes[node_id] = node
    return new_nodes, new_inputs


def compiled_additions(input_type, person_count):
    """Returns (new_nodes, new_inputs, class_types) for this workflow, or None if there is no compiled variant."""
    name = workflow_name(input_type, person_count)
    compiled_name = COMPILED_WORKFLOWS.get(name)
    if compiled_name is None:
        return None
    compiled = load_workflow(compiled_name)
    new_nodes, new_inputs = workflow_delta(load_workflow(name), compiled)
    classes = {node_id: compiled[node_id].get("class_type") for node_id in new_inputs}
    return new_nodes, new_inputs, classes


def apply_additions(prompt, additions):
    """Applies the compiled-variant additions. Raises ValueError (prompt untouched) if the queued
    prompt does not look like the workflow the additions were made for."""
    new_nodes, new_inputs, classes = additions
    for node_id, class_type in classes.items():
        node = prompt.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            raise ValueError(f"queued prompt has no {class_type} node {node_id}")
    for node_id in new_nodes:
        if node_id in prompt:
            raise ValueError(f"queued prompt already has node {node_id}")
    for node_id, node in new_nodes.items():
        prompt[node_id] = copy.deepcopy(node)
    for node_id, inputs in new_inputs.items():
        prompt[node_id].setdefault("inputs", {}).update(copy.deepcopy(inputs))


def apply_fp8_fast(prompt):
    """Returns a log message describing what happened."""
    inputs = prompt.get(LOADER_NODE, {}).get("inputs", {})
    current = inputs.get("quantization")
    if current != FP8_FROM:
        return f"[FP8] fast requested but node {LOADER_NODE} quantization is {current!r}, not {FP8_FROM!r}; unchanged"
    inputs["quantization"] = FP8_TO
    return f"[FP8] fast on: node {LOADER_NODE} quantization {FP8_FROM} -> {FP8_TO} (changes numerics, test only)"


def steps_per_window(prompt):
    """Sampler steps per InfiniteTalk window: steps - start_step (multitalk_loop runs len(timesteps)-1)."""
    try:
        inputs = prompt[SAMPLER_NODE]["inputs"]
        steps = int(inputs["steps"])
        start = int(inputs.get("start_step", 0) or 0)
        end = int(inputs.get("end_step", -1))
    except (KeyError, TypeError, ValueError):
        return None
    if end != -1:
        steps = min(steps, end)
    return max(1, steps - start)


_COMPILE_ERROR_MARKERS = ("torch._dynamo", "torch._inductor", "triton", "BackendCompilerFailed", "InductorError")


def is_compile_error(result):
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    text = f"{details.get('exception_type') or ''} {details.get('exception_message') or ''}"
    return any(marker in text for marker in _COMPILE_ERROR_MARKERS)


# ---------------------------------------------------------------- benchmark audio trim
def _trim_wav_bytes(data, seconds):
    """Sample-exact cut of a PCM WAV with the stdlib (no re-encoding). Raises wave.Error/EOFError otherwise."""
    with wave.open(io.BytesIO(data), "rb") as src:
        params = src.getparams()
        keep = min(src.getnframes(), int(src.getframerate() * seconds))
        frames = src.readframes(keep)
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return out.getvalue()


def _trim_other_bytes(data, seconds):
    """Any format librosa can read (mp3, flac, float WAV ...) -> 16-bit PCM WAV of the first `seconds`."""
    import librosa
    import soundfile

    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        y, sr = librosa.load(path, sr=None, mono=False, duration=seconds)
    finally:
        os.remove(path)
    out = io.BytesIO()
    soundfile.write(out, y.T if getattr(y, "ndim", 1) > 1 else y, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()


def trim_audio_b64(b64, seconds=BENCHMARK_SECONDS):
    data = base64.b64decode(b64)
    try:
        trimmed = _trim_wav_bytes(data, seconds)
    except (wave.Error, EOFError):
        trimmed = _trim_other_bytes(data, seconds)
    return base64.b64encode(trimmed).decode("ascii")


def benchmark_input(job_input):
    """Returns the job input with audio cut to BENCHMARK_SECONDS. Raises ValueError if it cannot."""
    if "wav_base64" not in job_input:
        raise ValueError("benchmark needs the audio as wav_base64 (wav_url / wav_path are not trimmed)")
    out = dict(job_input)
    for key in ("wav_base64", "wav_base64_2"):
        if key in out:
            out[key] = trim_audio_b64(out[key])
    if out.get("max_frame") is not None:
        out["max_frame"] = min(int(out["max_frame"]), BENCHMARK_MAX_FRAME)
    return out
