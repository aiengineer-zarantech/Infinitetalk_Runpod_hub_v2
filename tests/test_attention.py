# ==============================================================================
# v27 attention switch (worker_guard/attention.py): ATTENTION_MODE parsing, the sageattn self-test result from
# gpu_check.py, node 122 changes, and the whole job path through the REAL handler with ComfyUI mocked.
# ==============================================================================
import base64
import io
import json
import shutil
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

from worker_guard import attention

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "tests" / "fixtures" / "prompt_driver.py"
OK_INFO = {"name": "NVIDIA GeForce RTX 5090", "capability": "12.0", "sageattn": "ok"}
FAILED_INFO = {"name": "NVIDIA L40S", "capability": "8.9",
               "sageattn": "failed: AcceleratorError: CUDA error: no kernel image is available for execution"}


@pytest.fixture(autouse=True)
def no_attention_env(monkeypatch):
    monkeypatch.delenv("ATTENTION_MODE", raising=False)


@pytest.mark.parametrize("env, job, benchmark, expected", [
    (None, {}, False, ("sageattn", "default")),
    ("sdpa", {}, False, ("sdpa", "ATTENTION_MODE")),
    (" SDPA ", {}, False, ("sdpa", "ATTENTION_MODE")),
    ("sageattn", {}, False, ("sageattn", "ATTENTION_MODE")),
    ("", {}, False, ("sageattn", "default")),
    ("flash_attn_2", {}, False, ("sageattn", "default")),  # not offered: falls back to the default
    (None, {"attention_mode": "sdpa"}, False, ("sageattn", "default")),  # per job only for benchmark jobs
    (None, {"attention_mode": "sdpa"}, True, ("sdpa", "job input")),
    ("sdpa", {"attention_mode": "sageattn"}, True, ("sageattn", "job input")),
    ("sdpa", {"attention_mode": "nope"}, True, ("sdpa", "ATTENTION_MODE")),
])
def test_requested_mode(monkeypatch, env, job, benchmark, expected):
    if env is not None:
        monkeypatch.setenv("ATTENTION_MODE", env)
    assert attention.requested_mode(job, benchmark) == expected


def test_invalid_env_is_logged(monkeypatch, caplog):
    monkeypatch.setenv("ATTENTION_MODE", "flash_attn_2")
    attention.requested_mode({}, False)
    assert "invalid ATTENTION_MODE='flash_attn_2'" in caplog.text


@pytest.mark.parametrize("env, info, expected_mode, note_part", [
    (None, None, "sageattn", "sageattn (default, no self-test result)"),
    (None, OK_INFO, "sageattn", "sageattn (default, self-test ok)"),
    (None, FAILED_INFO, "sdpa", "sdpa (sageattn unavailable on NVIDIA L40S: failed: AcceleratorError"),
    ("sdpa", OK_INFO, "sdpa", "sdpa (ATTENTION_MODE)"),
    ("sdpa", FAILED_INFO, "sdpa", "sdpa (ATTENTION_MODE)"),
])
def test_decide(monkeypatch, env, info, expected_mode, note_part):
    if env:
        monkeypatch.setenv("ATTENTION_MODE", env)
    mode, note = attention.decide({}, False, info)
    assert mode == expected_mode and note.startswith(note_part), note


def _workflow(name):
    return json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["I2V_single.json", "I2V_multi.json", "V2V_single.json", "V2V_multi.json"])
def test_apply_changes_only_the_loader_attention(name):
    original = _workflow(name)
    prompt = _workflow(name)
    assert attention.apply(prompt, "sageattn") == [] and prompt == original
    assert attention.apply(prompt, "sdpa") == ["122"]
    assert prompt["122"]["inputs"]["attention_mode"] == "sdpa"
    prompt["122"]["inputs"]["attention_mode"] = "sageattn"
    assert prompt == original


# ---------------------------------------------------------------- self-test with a fake torch / sageattention
class FakeT:
    def __init__(self, shape=attention.SELF_TEST_SHAPE, finite=True):
        self.shape = shape
        self.finite = finite

    def float(self):
        return self

    def flatten(self):
        return self

    def all(self):
        return self

    def item(self):
        return self.finite


def fake_torch(cosine=0.9995, finite=True, shape=attention.SELF_TEST_SHAPE):
    gen = types.SimpleNamespace(manual_seed=lambda s: gen)
    functional = types.SimpleNamespace(
        scaled_dot_product_attention=lambda q, k, v: FakeT(),
        cosine_similarity=lambda a, b, dim: types.SimpleNamespace(item=lambda: cosine),
    )
    return types.SimpleNamespace(
        Generator=lambda device: gen, randn=lambda *a, **k: FakeT(), bfloat16="bf16",
        nn=types.SimpleNamespace(functional=functional), cuda=types.SimpleNamespace(synchronize=lambda i: None),
        isfinite=lambda t: FakeT(finite=finite), _out=FakeT(shape=shape),
    )


def install_sage(monkeypatch, torch, raise_error=None):
    module = types.ModuleType("sageattention")

    def sageattn(q, k, v, tensor_layout, is_causal):
        assert tensor_layout == "HND" and is_causal is False
        if raise_error:
            raise raise_error
        return torch._out

    module.sageattn = sageattn
    monkeypatch.setitem(sys.modules, "sageattention", module)


def test_self_test_ok(monkeypatch):
    torch = fake_torch()
    install_sage(monkeypatch, torch)
    assert attention.self_test(torch) == "ok"


@pytest.mark.parametrize("kwargs, error, expected", [
    ({}, RuntimeError("CUDA error: no kernel image is available for execution on the device\nmore"),
     "failed: RuntimeError: CUDA error: no kernel image is available for execution on the device"),
    ({"cosine": 0.5}, None, "failed: cosine similarity to sdpa 0.5000 < 0.99"),
    ({"finite": False}, None, "failed: non-finite output"),
    ({"shape": (1, 8, 512, 64)}, None, "failed: output shape (1, 8, 512, 64), expected (1, 8, 512, 128)"),
])
def test_self_test_failures_never_raise(monkeypatch, kwargs, error, expected):
    torch = fake_torch(**kwargs)
    install_sage(monkeypatch, torch, error)
    assert attention.self_test(torch) == expected


def test_self_test_without_sageattention(monkeypatch):
    monkeypatch.setitem(sys.modules, "sageattention", None)  # import raises ImportError
    assert attention.self_test(fake_torch()).startswith("failed: ModuleNotFoundError")


# ---------------------------------------------------------------- the whole job path, ComfyUI mocked
def _wav_b64(seconds=2, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


STUDIO = {"input_type": "image", "person_count": "single", "prompt": "A person talking naturally",
          "width": 854, "height": 480, "network_volume": False, "image_base64": "iVBORw0KGgo=",
          "wav_base64": _wav_b64()}


def _drive(tmp_path, jobs):
    root = tmp_path / "tree"
    root.mkdir()
    for name in ["handler.py", "setup_models.py", "worker_guard", "I2V_single.json", "I2V_multi.json",
                 "V2V_single.json", "V2V_multi.json", "I2V_single_compiled.json", "I2V_multi_compiled.json"]:
        src = REPO_ROOT / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, root / name)
    jobs_path, out = tmp_path / "jobs.json", tmp_path / "out.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(DRIVER), str(root), str(jobs_path), str(out)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8")), proc.stderr + proc.stdout


def test_job_path_falls_back_to_sdpa_only_where_sage_failed(tmp_path):
    ok, failed = tmp_path / "ok.json", tmp_path / "failed.json"
    ok.write_text(json.dumps(OK_INFO), encoding="utf-8")
    failed.write_text(json.dumps(FAILED_INFO), encoding="utf-8")
    result, log = _drive(tmp_path, [
        ({"WORKER_GPU_INFO": str(ok)}, STUDIO),
        ({"WORKER_GPU_INFO": str(failed)}, STUDIO),
        ({"WORKER_GPU_INFO": str(ok), "ATTENTION_MODE": "sdpa"}, STUDIO),
        ({"WORKER_GPU_INFO": str(ok)}, {**STUDIO, "benchmark": True, "attention_mode": "sdpa",
                                        "torch_compile": False}),
    ])
    default, fallback, env_sdpa, bench = result["prompts"]
    assert default["122"]["inputs"]["attention_mode"] == "sageattn"
    for prompt in (fallback, env_sdpa):
        assert prompt["122"]["inputs"]["attention_mode"] == "sdpa"
        prompt["122"]["inputs"]["attention_mode"] = "sageattn"
        assert prompt == default  # nothing else differs
    assert bench["122"]["inputs"]["attention_mode"] == "sdpa" and "compile_args" not in bench["122"]["inputs"]
    assert all(set(r) == {"video"} for r in result["results"])
    assert "[ACCEL] attention sageattn (default, self-test ok), torch_compile on, blocks_to_swap 0" in log
    assert "[ACCEL] attention sdpa (sageattn unavailable on NVIDIA L40S: failed: AcceleratorError" in log
    assert "[ACCEL] attention_mode sdpa set on node 122" in log
    assert "[ACCEL] attention sdpa (job input), torch_compile off (job input)" in log
