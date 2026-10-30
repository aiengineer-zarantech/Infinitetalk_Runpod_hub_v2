# ==============================================================================
# conftest.py - pytest setup for worker TTS tests (Windows/CPU friendly, no GPU)
# ==============================================================================
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mock modules that may only exist inside the RunPod container (installed before any test module imports handler.py)
for mod in ["runpod", "runpod.serverless", "websocket", "soundfile"]:
    if mod not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.serverless = MagicMock()
        mock_mod.serverless.start = MagicMock()
        sys.modules[mod] = mock_mod

if "librosa" not in sys.modules:
    mock_librosa = MagicMock()
    mock_effects = MagicMock()
    mock_librosa.effects = mock_effects
    sys.modules["librosa"] = mock_librosa
    sys.modules["librosa.effects"] = mock_effects

import pytest  # noqa: E402

HEALTHY_SYSTEM_STATS = {
    "system": {},
    "devices": [{
        "name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync", "type": "cuda", "index": 0,
        "vram_total": 32 * 1024 ** 3, "vram_free": 31 * 1024 ** 3,
    }],
}


@pytest.fixture(autouse=True)
def worker_accel_env(monkeypatch, tmp_path):
    """Compiled/original workflows come from the repo; no GPU info file; env defaults; fresh per-worker state."""
    from worker_guard import accel
    monkeypatch.setenv("WORKFLOW_DIR", str(REPO_ROOT))
    monkeypatch.setenv("WORKER_GPU_INFO", str(tmp_path / "no_worker_gpu.json"))
    for name in ("TORCH_COMPILE", "FP8_FAST", "OFFLOAD_TIERS", "OFFLOAD_480P_MAX_PIXELS", "OFFLOAD_CALIBRATION",
                 "OFFLOAD_TARGET_FRACTION", "RUNPOD_EXECUTION_TIMEOUT_SEC", "RUNPOD_RETURN_MARGIN_SEC",
                 "OOM_RETRY_MIN_SEC", "ALLOC_MODE", "VRAM_STAGES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(accel, "_compile_disabled_reason", None)


GB = 1024 ** 3
# What the fake container reports: 64 GB limit, 8 GB used
FAKE_RAM = {"limit": 64 * GB, "used": 8 * GB, "available": 56 * GB, "source": "cgroup v2"}


@pytest.fixture(autouse=True)
def fresh_worker_state(monkeypatch):
    """Per-worker state of job_guard (job counter, resident model, VRAM held by our models) starts empty,
    the container RAM is a fixed fake, and /worker/vram has no data unless a test sets it."""
    from worker_guard import comfy_api, job_guard, sysmem
    monkeypatch.setattr(job_guard, "_jobs_on_worker", 0)
    monkeypatch.setattr(job_guard, "_loader_sig", None)
    monkeypatch.setattr(job_guard, "_resident_used", 0)
    monkeypatch.setattr(sysmem, "read_ram", lambda read=None: dict(FAKE_RAM))

    def no_vram_data(timeout=5):
        raise ConnectionError("no /worker/vram in tests")

    monkeypatch.setattr(comfy_api, "vram_stages", no_vram_data)


@pytest.fixture(autouse=True)
def healthy_comfy_gpu(monkeypatch):
    """handler.py installs worker_guard, which calls ComfyUI /system_stats before every job.
    Every test gets a mocked healthy cuda device unless it patches comfy_api itself. No network."""
    from worker_guard import comfy_api
    monkeypatch.setattr(comfy_api, "system_stats", lambda timeout=10: HEALTHY_SYSTEM_STATS)
    monkeypatch.setattr(comfy_api, "free", lambda unload_models=True, free_memory=True, timeout=10: None)
