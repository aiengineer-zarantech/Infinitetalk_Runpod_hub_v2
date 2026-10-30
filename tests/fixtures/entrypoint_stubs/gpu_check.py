# Stub for worker_guard/gpu_check.py in the entrypoint smoke test
import os
import sys

if os.environ.get("STUB_GPU_FAIL"):
    print("[GPU] FAILED: stub failure", flush=True)
    sys.exit(1)
print("[GPU] OK: Stub GPU, 24.0 GB, CUDA 12.8", flush=True)
