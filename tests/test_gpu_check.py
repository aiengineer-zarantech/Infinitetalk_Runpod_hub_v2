# ==============================================================================
# Startup GPU guard (worker_guard/gpu_check.py) with a fake torch module. No GPU needed.
# ==============================================================================
import sys
import types

import pytest

from worker_guard import gpu_check


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, value, finite=True):
        self.value = value
        self.finite = finite

    def __matmul__(self, other):
        return FakeTensor(self.value * other.value * gpu_check.MATMUL_SIZE, self.finite and other.finite)

    def __eq__(self, other):
        return FakeScalar(self.value == other)


def make_torch(available=True, count=1, matmul_value=1.0, finite=True, raise_on_ones=None,
               name="NVIDIA GeForce RTX 5090", total=32 * 1024 ** 3, cuda_version="12.8"):
    torch = types.ModuleType("torch")
    props = types.SimpleNamespace(name=name, total_memory=total, major=8, minor=9)
    synced = []

    def ones(shape, device=None, dtype=None):
        if raise_on_ones:
            raise raise_on_ones
        assert device == "cuda:0"
        return FakeTensor(matmul_value, finite)

    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        get_device_properties=lambda i: props,
        synchronize=lambda i=None: synced.append(i),
    )
    torch.ones = ones
    torch.float32 = "float32"
    torch.isfinite = lambda t: FakeScalar(t.finite)
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    torch._synced = synced
    return torch


def test_run_check_ok_reports_device_and_synchronizes():
    torch = make_torch()
    info = gpu_check.run_check(torch)
    assert info["name"] == "NVIDIA GeForce RTX 5090"
    assert info["total_gb"] == pytest.approx(32.0)
    assert info["cuda"] == "12.8"
    assert torch._synced == [0]


@pytest.mark.parametrize("kwargs, reason", [
    ({"available": False}, "is_available() returned False"),
    ({"count": 0}, "device_count() returned 0"),
    ({"matmul_value": 0.5}, "wrong values"),
    ({"finite": False}, "wrong values"),
])
def test_run_check_failures(kwargs, reason):
    with pytest.raises(gpu_check.GpuCheckError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        gpu_check.run_check(make_torch(**kwargs))


def run_main(monkeypatch, capsys, torch):
    monkeypatch.setitem(sys.modules, "torch", torch)
    code = gpu_check.main()
    return code, capsys.readouterr().out


def test_main_ok_line_format(monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, make_torch())
    assert code == 0
    assert out.splitlines()[0] == "[GPU] OK: NVIDIA GeForce RTX 5090, 32.0 GB, CUDA 12.8"
    assert "matmul OK" in out


@pytest.mark.parametrize("total_gb, env, ok", [
    (31.4, None, True), (28.0, None, True), (27.99, None, False), (23.6, None, False), (31.4, "40", False),
])
def test_main_rejects_small_gpu_at_startup(monkeypatch, capsys, tmp_path, total_gb, env, ok):
    monkeypatch.setattr(gpu_check, "GPU_INFO_PATH", str(tmp_path / "gpu.json"))
    if env:
        monkeypatch.setenv("MIN_VRAM_GB", env)
    code, out = run_main(monkeypatch, capsys, make_torch(name="NVIDIA GeForce RTX 4090", total=int(total_gb * 1024 ** 3)))
    assert (code == 0) is ok
    if not ok:
        minimum = env or "28"
        assert out.strip() == f"[GPU] FAILED: NVIDIA GeForce RTX 4090 has {total_gb:.1f} GB VRAM, minimum is {minimum} GB"
        assert not (tmp_path / "gpu.json").exists()


def test_main_failed_when_cuda_unavailable(monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, make_torch(available=False))
    assert code == 1
    assert out.strip() == "[GPU] FAILED: GpuCheckError: torch.cuda.is_available() returned False"


def test_main_failed_on_cuda_runtime_error(monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, make_torch(raise_on_ones=RuntimeError("CUDA error: no kernel image")))
    assert code == 1
    assert out.startswith("[GPU] FAILED: RuntimeError: CUDA error: no kernel image")


def test_main_failed_when_torch_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    code = gpu_check.main()
    assert code == 1
    assert capsys.readouterr().out.startswith("[GPU] FAILED: ")


def test_main_failed_on_hang(monkeypatch, capsys):
    import threading
    release = threading.Event()
    torch = make_torch()
    torch.cuda.is_available = lambda: release.wait(5) or True
    monkeypatch.setattr(gpu_check, "GPU_CHECK_TIMEOUT_SEC", 0.2)
    try:
        code, out = run_main(monkeypatch, capsys, torch)
    finally:
        release.set()
    assert code == 1
    assert "[GPU] FAILED: check did not finish within" in out
