#!/usr/bin/env python3
"""Startup GPU guard. Runs before any model copy or ComfyUI start.

Exit code 0: the GPU is usable. Exit code 1: it is not, so the container stops and
RunPod restarts the worker. Must run with the same python as ComfyUI.
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker_guard.offload import min_vram_gb  # noqa: E402  (stdlib only; same MIN_VRAM_GB as the job check)

MATMUL_SIZE = 1024
GPU_CHECK_TIMEOUT_SEC = float(os.environ.get("GPU_CHECK_TIMEOUT_SEC", "120"))
# Read by the handler process (worker_guard.accel) to decide whether torch.compile can work
GPU_INFO_PATH = os.environ.get("WORKER_GPU_INFO", "/tmp/worker_gpu.json")


class GpuCheckError(Exception):
    pass


def run_check(torch):
    """Raises GpuCheckError (or any torch error) if cuda:0 is not usable, else returns device info."""
    if not torch.cuda.is_available():
        raise GpuCheckError("torch.cuda.is_available() returned False")
    count = torch.cuda.device_count()
    if count < 1:
        raise GpuCheckError(f"torch.cuda.device_count() returned {count}")
    props = torch.cuda.get_device_properties(0)

    # Small matmul on cuda:0: every element of ones(N, N) @ ones(N, N) must equal N
    start = time.time()
    a = torch.ones((MATMUL_SIZE, MATMUL_SIZE), device="cuda:0", dtype=torch.float32)
    b = a @ a
    torch.cuda.synchronize(0)
    if not bool(torch.isfinite(b).all().item()) or not bool((b == float(MATMUL_SIZE)).all().item()):
        raise GpuCheckError("matmul on cuda:0 returned wrong values")
    return {
        "name": props.name,
        "total_gb": props.total_memory / (1024 ** 3),
        "cuda": torch.version.cuda,
        "count": count,
        "capability": f"{props.major}.{props.minor}",
        "matmul_ms": (time.time() - start) * 1000,
    }


def main():
    result = {}

    def target():
        try:
            import torch
            result["info"] = run_check(torch)
        except BaseException as e:
            result["error"] = f"{type(e).__name__}: {e}"

    # A hung driver must not hang startup forever
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(GPU_CHECK_TIMEOUT_SEC)

    if worker.is_alive():
        error = f"check did not finish within {GPU_CHECK_TIMEOUT_SEC:.0f}s (GPU or driver hang?)"
    else:
        error = result.get("error")
    if error or "info" not in result:
        print(f"[GPU] FAILED: {error or 'no result'}", flush=True)
        return 1

    info = result["info"]
    minimum = min_vram_gb()
    if info["total_gb"] < minimum:
        print(f"[GPU] FAILED: {info['name']} has {info['total_gb']:.1f} GB VRAM, minimum is {minimum:.0f} GB", flush=True)
        return 1
    print(f"[GPU] OK: {info['name']}, {info['total_gb']:.1f} GB, CUDA {info['cuda']}", flush=True)
    print(
        f"[GPU] devices={info['count']}, compute capability {info['capability']}, "
        f"{MATMUL_SIZE}x{MATMUL_SIZE} matmul OK in {info['matmul_ms']:.0f} ms",
        flush=True,
    )
    try:
        with open(GPU_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump({"name": info["name"], "capability": info["capability"],
                       "total_gb": round(info["total_gb"], 2), "cuda": info["cuda"]}, f)
    except OSError as e:
        print(f"[GPU] could not write {GPU_INFO_PATH}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Skip interpreter teardown so a stuck CUDA thread cannot block the exit
    os._exit(code)
