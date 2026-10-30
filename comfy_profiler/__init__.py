"""zz_worker_profiler: a ComfyUI custom node package that defines NO nodes.

Copied by Dockerfile.v22 to /ComfyUI/custom_nodes/zz_worker_profiler. With PROFILE=1 in the
worker environment it times the InfiniteTalk loop per window (see profiler.py); otherwise it
imports nothing and changes nothing.
"""
import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _enabled():
    return os.environ.get("PROFILE", "").strip().lower() in ("1", "true", "yes", "on")


if _enabled():
    try:
        from . import profiler
        profiler.register()
    except Exception as e:
        logging.getLogger("worker_profiler").warning(f"[PROFILE] not enabled: {type(e).__name__}: {e}")
