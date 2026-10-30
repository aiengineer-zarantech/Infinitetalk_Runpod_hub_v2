# Stub for handler.py in the entrypoint smoke test: report, then stop the stub ComfyUI
import os
import urllib.request

print(f"HANDLER STARTED cwd={os.getcwd()}", flush=True)
for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE"):
    print(f"HANDLER ENV {name}={os.environ.get(name, '<unset>')}", flush=True)
try:
    urllib.request.urlopen(f"http://127.0.0.1:{os.environ['COMFY_PORT']}/__stop", timeout=5).read()
except Exception:
    pass
