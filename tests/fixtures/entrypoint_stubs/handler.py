# Stub for handler.py in the entrypoint smoke test: report, then stop the stub ComfyUI
import os
import urllib.request

print(f"HANDLER STARTED cwd={os.getcwd()}", flush=True)
try:
    urllib.request.urlopen(f"http://127.0.0.1:{os.environ['COMFY_PORT']}/__stop", timeout=5).read()
except Exception:
    pass
