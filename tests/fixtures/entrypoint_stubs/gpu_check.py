# Stub for worker_guard/gpu_check.py in the entrypoint smoke test
import json
import os
import sys

if os.environ.get("STUB_GPU_FAIL"):
    print("[GPU] FAILED: stub failure", flush=True)
    sys.exit(1)
print("[GPU] OK: Stub GPU, 24.0 GB, CUDA 12.8", flush=True)
# v27: the real check also writes the sageattn self-test result to the GPU info file
if os.environ.get("STUB_SAGE"):
    with open(os.environ["WORKER_GPU_INFO"], "w", encoding="utf-8") as f:
        json.dump({"name": "Stub GPU", "sageattn": os.environ["STUB_SAGE"]}, f)
