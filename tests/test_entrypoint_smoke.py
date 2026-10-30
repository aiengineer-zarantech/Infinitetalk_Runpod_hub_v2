# ==============================================================================
# entrypoint.sh smoke test: runs the REAL script in bash (Git Bash on Windows) with stub
# GPU check, model copy, ComfyUI and handler, and the REAL worker_guard/startup_check.py.
# Checks ordering (copy and ComfyUI in parallel, handler last), [comfy] log forwarding,
# the [STARTUP] line, and every failure exit path.
# ==============================================================================
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUBS = REPO_ROOT / "tests" / "fixtures" / "entrypoint_stubs"


def find_bash():
    if os.environ.get("WORKER_TEST_BASH"):
        return os.environ["WORKER_TEST_BASH"]
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            for rel in ("bin/bash.exe", "usr/bin/bash.exe"):
                candidate = Path(git).resolve().parent.parent / rel
                if candidate.exists():
                    return str(candidate)
        return None  # never fall back to the WSL bash.exe
    return shutil.which("bash")


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="Git Bash / bash not found")


def posix(path):
    return str(path).replace("\\", "/")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "root"
    (root / "worker_guard").mkdir(parents=True)
    # Same as the image (COPY worker_guard/), with gpu_check.py replaced by a stub
    for src in (REPO_ROOT / "worker_guard").glob("*.py"):
        shutil.copy(src, root / "worker_guard" / src.name)
    shutil.copy(STUBS / "gpu_check.py", root / "worker_guard" / "gpu_check.py")
    shutil.copy(STUBS / "setup_models.py", root / "setup_models.py")
    shutil.copy(STUBS / "handler.py", root / "handler.py")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    shutil.copy(STUBS / "comfy_main.py", comfy / "main.py")
    (comfy / "extra_model_paths.yaml").write_text("")

    def _run(**stub_env):
        env = os.environ.copy()
        env.update(
            WORKER_ROOT=posix(root), COMFY_DIR=posix(comfy),
            PYTHON_BIN=posix(sys.executable), PYTHON3_BIN=posix(sys.executable),
            COMFY_PORT=str(free_port()), COMFY_START_TIMEOUT="20", STUB_COMFY_LIFETIME="15",
        )
        env.update({k: str(v) for k, v in stub_env.items()})
        out_path = tmp_path / "out.txt"
        with open(out_path, "wb") as out:
            proc = subprocess.run([BASH, posix(REPO_ROOT / "entrypoint.sh")], stdout=out,
                                  stderr=subprocess.STDOUT, env=env, timeout=120)
        return proc.returncode, out_path.read_bytes().decode("utf-8", errors="replace"), comfy

    return _run


def index(out, text):
    pos = out.find(text)
    assert pos >= 0, f"{text!r} not in output:\n{out}"
    return pos


def test_happy_path_order_and_logs(run):
    rc, out, comfy = run()
    assert rc == 0, out
    gpu = index(out, "[GPU] OK: Stub GPU, 24.0 GB, CUDA 12.8")
    comfy_started = index(out, "[comfy] COMFY STUB STARTED")
    copy_done = index(out, "COPY STUB DONE")
    startup = index(out, "[STARTUP] ready in ")
    handler = index(out, "HANDLER STARTED")
    # GPU first, ComfyUI starts while the copy is still running, handler last
    assert gpu < comfy_started < copy_done < startup < handler
    assert gpu < index(out, "COPY STUB START")
    assert "[comfy] comfy stderr line" in out
    assert re.search(r"\[STARTUP\] ready in \d+\.\ds \(gpu_check \d+\.\ds, copy \d+\.\ds, comfy \d+\.\ds\)", out)
    assert "[COMFY] device cuda, cuda:0 Stub GPU : cudaMallocAsync, VRAM used 0.6 / 24.0 GB" in out
    assert "[STARTUP] idle VRAM used 600 MB" in out
    args_line = out[comfy_started:out.index("\n", comfy_started)]
    assert "--use-sage-attention" in args_line and "--cpu" not in args_line
    assert "--listen 127.0.0.1" in args_line
    # comfyui.log keeps the raw lines (the handler reads its tail on errors)
    log = (comfy / "comfyui.log").read_text(errors="replace")
    assert "COMFY STUB STARTED" in log and "[comfy]" not in log
    assert f"HANDLER STARTED cwd=" in out


def test_gpu_failure_exits_before_copy_and_comfy(run):
    rc, out, _ = run(STUB_GPU_FAIL=1)
    assert rc == 1
    assert "[GPU] FAILED: stub failure" in out
    assert "COPY STUB START" not in out and "COMFY STUB" not in out and "HANDLER STARTED" not in out


def test_copy_failure_exits(run):
    rc, out, _ = run(STUB_COPY_RC=2, STUB_COPY_SECONDS=1)
    assert rc == 1
    assert "setup_models.py failed (exit code 2)!" in out
    assert "HANDLER STARTED" not in out


def test_comfy_crash_exits(run):
    rc, out, _ = run(STUB_COMFY_CRASH=1, STUB_COPY_SECONDS=5)
    assert rc == 1
    assert "ComfyUI process crashed unexpectedly!" in out
    assert "[comfy] COMFY STUB STARTED" in out
    assert "HANDLER STARTED" not in out


def test_comfy_cpu_device_exits(run):
    rc, out, _ = run(STUB_COMFY_DEVICE="cpu", STUB_COPY_SECONDS=0)
    assert rc == 1
    assert "[COMFY] FAILED: device type is 'cpu', expected cuda" in out
    assert "[STARTUP] ready" not in out and "HANDLER STARTED" not in out


def test_comfy_start_timeout_exits(run):
    rc, out, _ = run(STUB_COMFY_DELAY=30, COMFY_START_TIMEOUT=3, STUB_COPY_SECONDS=0, STUB_COMFY_LIFETIME=1)
    assert rc == 1
    assert "ComfyUI failed to start within 3 seconds" in out
    assert "HANDLER STARTED" not in out
