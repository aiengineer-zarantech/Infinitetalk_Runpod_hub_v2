# Stub for setup_models.py in the entrypoint smoke test
import os
import sys
import time

print("COPY STUB START", flush=True)
time.sleep(float(os.environ.get("STUB_COPY_SECONDS", "2")))
print("COPY STUB DONE", flush=True)
sys.exit(int(os.environ.get("STUB_COPY_RC", "0")))
