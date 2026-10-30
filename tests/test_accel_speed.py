# ==============================================================================
# worker_guard/accel.py (compile/fp8/benchmark helpers), worker_guard/speed.py ([SPEED] math),
# scripts/make_compiled_workflows.py, scripts/verify_wrapper_contract.py, gpu_check GPU info file.
# ==============================================================================
import base64
import io
import json
import logging
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from worker_guard import accel, gpu_check
from worker_guard.speed import SpeedTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_compiled_workflows  # noqa: E402
import verify_wrapper_contract  # noqa: E402


# ---------------------------------------------------------------- compiled workflows
@pytest.mark.parametrize("original, compiled", list(accel.COMPILED_WORKFLOWS.items()))
def test_compiled_file_is_original_plus_one_link(original, compiled):
    a = json.loads((REPO_ROOT / original).read_text(encoding="utf-8"))
    b = json.loads((REPO_ROOT / compiled).read_text(encoding="utf-8"))
    assert accel.workflow_delta(a, b) == ({}, {"122": {"compile_args": ["177", 0]}})
    assert b["177"]["class_type"] == "WanVideoTorchCompileSettings"
    assert b["122"]["class_type"] == "WanVideoModelLoader"
    # sampler settings identical (steps, cfg, shift, scheduler, seed ...)
    assert a["128"] == b["128"] and a["138"] == b["138"] and a["192"] == b["192"]


def test_compiled_files_are_up_to_date():
    assert make_compiled_workflows.main(["--check"]) == 0


def test_originals_never_contain_compile_args():
    for name in ("I2V_single.json", "I2V_multi.json", "V2V_single.json", "V2V_multi.json"):
        wf = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        assert "compile_args" not in wf["122"]["inputs"], name


@pytest.mark.parametrize("mutate, message", [
    (lambda w: w["128"]["inputs"].__setitem__("steps", 4), "changes input 128.steps"),
    (lambda w: w["122"]["inputs"].pop("lora"), "removes input 122.lora"),
    (lambda w: w.pop("138"), "removes node 138"),
    (lambda w: w["177"].__setitem__("class_type", "Other"), "changes node 177 field"),
])
def test_workflow_delta_rejects_anything_but_additions(mutate, message):
    original = json.loads((REPO_ROOT / "I2V_single.json").read_text(encoding="utf-8"))
    variant = json.loads(json.dumps(original))
    mutate(variant)
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        accel.workflow_delta(original, variant)


def test_apply_additions_refuses_unexpected_prompt():
    additions = accel.compiled_additions("image", "single")
    with pytest.raises(ValueError, match="no WanVideoModelLoader node 122"):
        accel.apply_additions({"128": {"class_type": "WanVideoSampler", "inputs": {}}}, additions)


def test_workflow_name_mapping_matches_handler():
    import handler
    for it in ("image", "video"):
        for pc in ("single", "multi"):
            assert "/" + accel.workflow_name(it, pc) == handler.get_workflow_path(it, pc)


# ---------------------------------------------------------------- decisions
def write_gpu_info(tmp_path, monkeypatch, capability):
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"name": "x", "capability": capability}))
    monkeypatch.setenv("WORKER_GPU_INFO", str(path))


@pytest.mark.parametrize("env, job, benchmark, capability, expected", [
    (None, {}, False, None, True),                       # default on
    ("0", {}, False, None, False),
    ("1", {"torch_compile": False}, False, None, True),   # per-job ignored without benchmark
    ("1", {"torch_compile": False}, True, None, False),   # honoured for benchmark
    ("0", {"torch_compile": True}, True, None, True),
    (None, {}, False, "12.0", True),                      # RTX 5090 (Blackwell)
    (None, {}, False, "8.9", True),                       # RTX 4090
    (None, {}, False, "8.6", False),                      # A40 / 3090
    (None, {}, False, "8.0", False),                      # A100
])
def test_compile_decision(monkeypatch, tmp_path, env, job, benchmark, capability, expected):
    if env is not None:
        monkeypatch.setenv("TORCH_COMPILE", env)
    if capability:
        write_gpu_info(tmp_path, monkeypatch, capability)
    on, note = accel.compile_decision(job, benchmark)
    assert on is expected, note


def test_compile_disabled_after_failure(monkeypatch):
    accel.disable_compile("compile failed earlier on this worker")
    assert accel.compile_decision({}, False) == (False, "off (compile failed earlier on this worker)")


@pytest.mark.parametrize("env, job, benchmark, expected", [
    (None, {}, False, False), ("1", {}, False, True), ("0", {"fp8_fast": True}, False, False),
    ("0", {"fp8_fast": True}, True, True),
])
def test_fp8_decision_default_off(monkeypatch, env, job, benchmark, expected):
    if env is not None:
        monkeypatch.setenv("FP8_FAST", env)
    assert accel.fp8_decision(job, benchmark) is expected


def test_apply_fp8_fast():
    prompt = {"122": {"inputs": {"quantization": "fp8_e4m3fn"}}}
    assert "fp8_e4m3fn -> fp8_e4m3fn_fast" in accel.apply_fp8_fast(prompt)
    assert prompt["122"]["inputs"]["quantization"] == "fp8_e4m3fn_fast"
    other = {"122": {"inputs": {"quantization": "disabled"}}}
    assert "unchanged" in accel.apply_fp8_fast(other)
    assert other["122"]["inputs"]["quantization"] == "disabled"


@pytest.mark.parametrize("details, expected", [
    ({"exception_type": "torch._dynamo.exc.BackendCompilerFailed", "exception_message": "x"}, True),
    ({"exception_type": "torch._inductor.exc.InductorError", "exception_message": ""}, True),
    ({"exception_type": "triton.compiler.errors.CompilationError", "exception_message": ""}, True),
    ({"exception_type": "RuntimeError", "exception_message": "backend='inductor' raised: InductorError"}, True),
    ({"exception_type": "torch.OutOfMemoryError", "exception_message": "Allocation on device"}, False),
    ({"exception_type": "ValueError", "exception_message": "bad"}, False),
])
def test_is_compile_error(details, expected):
    assert accel.is_compile_error({"error": "x", "details": details}) is expected
    assert accel.is_compile_error({"error": "torch._dynamo"}) is False  # no details: not judged


@pytest.mark.parametrize("inputs, expected", [
    ({"steps": 6, "start_step": 0, "end_step": -1}, 6),
    ({"steps": 4, "start_step": 2, "end_step": -1}, 2),
    ({"steps": 6, "start_step": 0, "end_step": 4}, 4),
])
def test_steps_per_window(inputs, expected):
    assert accel.steps_per_window({"128": {"inputs": inputs}}) == expected


# ---------------------------------------------------------------- benchmark trim
def wav_bytes(seconds, rate=16000, channels=2, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(bytes(range(256)) * (int(rate * seconds) * channels * width // 256 + 1))
    return buf.getvalue()


def test_trim_wav_is_sample_exact():
    original = wav_bytes(30)
    trimmed = base64.b64decode(accel.trim_audio_b64(base64.b64encode(original).decode()))
    with wave.open(io.BytesIO(original)) as a, wave.open(io.BytesIO(trimmed)) as b:
        assert b.getparams()[:3] == a.getparams()[:3]
        assert b.getnframes() == 10 * 16000
        assert b.readframes(b.getnframes()) == a.readframes(10 * 16000)


def test_trim_short_audio_unchanged_length():
    original = wav_bytes(3)
    trimmed = base64.b64decode(accel.trim_audio_b64(base64.b64encode(original).decode()))
    with wave.open(io.BytesIO(original)) as a, wave.open(io.BytesIO(trimmed)) as b:
        assert a.getnframes() < 10 * 16000
        assert b.getnframes() == a.getnframes()
        assert b.readframes(b.getnframes()) == a.readframes(a.getnframes())


def test_trim_non_wav_uses_librosa_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(accel, "_trim_other_bytes", lambda data, seconds: calls.append((data, seconds)) or wav_bytes(1))
    accel.trim_audio_b64(base64.b64encode(b"ID3 fake mp3").decode())
    assert calls == [(b"ID3 fake mp3", 10)]


def test_benchmark_input_rules():
    b64 = base64.b64encode(wav_bytes(12)).decode()
    out = accel.benchmark_input({"wav_base64": b64, "wav_base64_2": b64, "max_frame": 999})
    assert out["max_frame"] == 331
    assert out["wav_base64"] != b64 and out["wav_base64_2"] != b64
    assert "max_frame" not in accel.benchmark_input({"wav_base64": b64})
    with pytest.raises(ValueError, match="needs the audio as wav_base64"):
        accel.benchmark_input({"wav_url": "https://example.com/a.wav"})


# ---------------------------------------------------------------- [SPEED]
def feed(tracker, events):
    for e in events:
        if e[0] == "exec":
            tracker.on_executing(e[1], e[2])
        else:
            tracker.on_progress("128", e[1], e[2], e[3])


def test_speed_windows_and_summary(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    t = SpeedTracker(steps_per_window=3)
    events = [("exec", "128", 100.0)]
    # window 1: compile/model load delay 40s, then 5 s/it; window 2 starts after a 10s VAE decode
    times = [145.0, 150.0, 155.0, 170.0, 175.0, 180.0]
    events += [("prog", i + 1, 6, ts) for i, ts in enumerate(times)]
    events += [("exec", "131", 190.0)]
    feed(t, events)
    assert caplog.messages == ["[SPEED] window 1/2: 55.0s, 5.00 s/it", "[SPEED] window 2/2: 25.0s, 5.00 s/it"]
    assert t.avg_s_per_it() == pytest.approx(5.0)
    assert t.sampling_sec() == pytest.approx(90.0)
    assert t.first_step_overhead() == pytest.approx(40.0)


def test_speed_ignores_other_nodes_and_warns_once(caplog):
    caplog.set_level(logging.WARNING, logger="worker_guard")
    t = SpeedTracker(steps_per_window=6)
    t.on_progress("301", 1, 10, 1.0)  # MelBandRoFormer progress
    assert t.step_times == {}
    t.on_progress("128", 1, 10, 2.0)
    t.on_progress("128", 2, 10, 3.0)
    assert len([m for m in caplog.messages if "ignoring node 128 progress with max 10" in m]) == 1
    assert t.windows == []


def _loading_then_sampling(t, loading_max):
    """Real node 128 sequence: 'Loading transformer parameters' (max 1633), then sampling (max 126 =
    21 windows x 6 steps for a 60 s job). Sampling: 5 s/it, 15 s between windows (VAE decode)."""
    t.on_executing("128", 0.0)
    for v in range(1, loading_max + 1):
        t.on_progress("128", v, loading_max, v * 0.01)
    for w in range(21):
        for s in range(6):
            t.on_progress("128", w * 6 + s + 1, 126, 100.0 + w * 40.0 + s * 5.0)
    t.on_executing("131", 935.0)


def test_speed_ignores_model_loading_progress_max_1633_then_126(caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    t = SpeedTracker(steps_per_window=6)
    _loading_then_sampling(t, 1633)
    speed = [m for m in caplog.messages if m.startswith("[SPEED] window")]
    assert len(speed) == 21
    assert speed[0] == "[SPEED] window 1/21: 125.0s, 5.00 s/it"  # from sampler start, includes loading
    assert speed[1] == "[SPEED] window 2/21: 40.0s, 5.00 s/it"
    assert speed[-1] == "[SPEED] window 21/21: 40.0s, 5.00 s/it"
    assert not any("/1633" in m or "/272" in m for m in caplog.messages)
    assert [m for m in caplog.messages if "ignoring node 128 progress with max 1633" in m] != []
    assert t.avg_s_per_it() == pytest.approx(5.0)
    assert t.sampling_sec() == pytest.approx(935.0)
    # first sampling step at t=100, not the first loading message at t=0.01
    assert t.first_step_overhead() == pytest.approx(95.0)


def test_speed_discards_phase_when_max_changes(caplog):
    # If a loading max happened to be a multiple of 6 (1632 = 272 x 6), the max change to 126 discards it
    caplog.set_level(logging.INFO, logger="worker_guard")
    t = SpeedTracker(steps_per_window=6)
    _loading_then_sampling(t, 1632)
    assert "[SPEED] progress max changed 1632 -> 126; discarding 272 window(s)" in caplog.messages
    assert [w[0] for w in t.windows] == list(range(1, 22))
    assert t.avg_s_per_it() == pytest.approx(5.0)
    assert t.first_step_overhead() == pytest.approx(95.0)


def test_speed_single_step_windows():
    t = SpeedTracker(steps_per_window=1)
    feed(t, [("exec", "128", 0.0), ("prog", 1, 2, 4.0), ("prog", 2, 2, 7.0)])
    assert [w[2] for w in t.windows] == [4.0, 3.0]


# ---------------------------------------------------------------- verify_wrapper_contract.py
WRAPPER_SRC = '''
class WanVideoTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "dynamo_cache_size_limit": ("INT", {"default": 64}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {"dynamo_recompile_limit": ("INT", {"default": 128})},
        }
    RETURN_TYPES = ("WANCOMPILEARGS",)

class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {}),
                "quantization": (["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast"], {"default": "disabled"}),
            },
            "optional": {"compile_args": ("WANCOMPILEARGS", )},
        }
'''


def run_contract(tmp_path, source):
    (tmp_path / "nodes_model_loading.py").write_text(source, encoding="utf-8")
    pairs = [(str(REPO_ROOT / o), str(REPO_ROOT / c)) for o, c in accel.COMPILED_WORKFLOWS.items()]
    return verify_wrapper_contract.check(str(tmp_path), pairs)


def test_contract_ok(tmp_path):
    assert run_contract(tmp_path, WRAPPER_SRC) == []


@pytest.mark.parametrize("old, new, expected", [
    ('"backend"', '"compile_backend"', "input 'backend' is not in the installed INPUT_TYPES"),
    ('["inductor","cudagraphs"]', '["cudagraphs"]', "backend='inductor' not in installed options"),
    ('"optional": {"compile_args": ("WANCOMPILEARGS", )}', '"optional": {}', "no optional compile_args"),
    ('"fp8_e4m3fn", "fp8_e4m3fn_fast"]', '"fp8_e4m3fn"]', "no option 'fp8_e4m3fn_fast'"),
    ('RETURN_TYPES = ("WANCOMPILEARGS",)', 'RETURN_TYPES = ("OTHER",)', "RETURN_TYPES"),
    ('"dynamic": ("BOOLEAN", {"default": False}),', '"dynamic": ("BOOLEAN", {"default": False}),\n                "new_required": ("INT", {}),', "misses required inputs ['new_required']"),
])
def test_contract_detects_mismatch(tmp_path, old, new, expected):
    errors = run_contract(tmp_path, WRAPPER_SRC.replace(old, new))
    assert any(expected in e for e in errors), errors


def test_contract_cli_separator(tmp_path):
    (tmp_path / "nodes_model_loading.py").write_text(WRAPPER_SRC, encoding="utf-8")
    args = [str(tmp_path)] + [f"{REPO_ROOT / o}={REPO_ROOT / c}" for o, c in accel.COMPILED_WORKFLOWS.items()]
    assert verify_wrapper_contract.main(args) == 0


# ---------------------------------------------------------------- gpu_check writes the info file
def test_gpu_check_writes_info_file(monkeypatch, capsys, tmp_path):
    from tests.test_gpu_check import make_torch
    info_path = tmp_path / "worker_gpu.json"
    monkeypatch.setattr(gpu_check, "GPU_INFO_PATH", str(info_path))
    monkeypatch.setitem(sys.modules, "torch", make_torch(name="NVIDIA GeForce RTX 5090", total=int(31.4 * 1024 ** 3)))
    assert gpu_check.main() == 0
    info = json.loads(info_path.read_text())
    assert info["name"] == "NVIDIA GeForce RTX 5090" and info["capability"] == "8.9"
    monkeypatch.setenv("WORKER_GPU_INFO", str(info_path))
    assert accel.gpu_info()["total_gb"] == 31.4
