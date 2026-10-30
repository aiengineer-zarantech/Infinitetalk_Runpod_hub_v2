# Stub for ComfyUI main.py in the entrypoint smoke test (copied to <COMFY_DIR>/main.py)
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()

print("COMFY STUB STARTED args=" + " ".join(sys.argv[1:]))
print("comfy stderr line", file=sys.stderr)
if os.environ.get("STUB_COMFY_CRASH"):
    sys.exit(3)
time.sleep(float(os.environ.get("STUB_COMFY_DELAY", "0")))

STATS = {"system": {"argv": ["main.py", *sys.argv[1:]]}, "devices": [{
    "name": "cuda:0 Stub GPU : cudaMallocAsync" if os.environ.get("STUB_COMFY_DEVICE", "cuda") == "cuda" else "cpu",
    "type": os.environ.get("STUB_COMFY_DEVICE", "cuda"), "index": 0,
    "vram_total": 24 * 1024 ** 3, "vram_free": 24 * 1024 ** 3 - 600 * 1024 ** 2,
}]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(STATS).encode() if self.path == "/system_stats" else b"ok"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
        if self.path == "/__stop":
            threading.Thread(target=server.shutdown).start()

    def log_message(self, *a):
        pass


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
threading.Timer(float(os.environ.get("STUB_COMFY_LIFETIME", "20")), server.shutdown).start()
server.serve_forever()
