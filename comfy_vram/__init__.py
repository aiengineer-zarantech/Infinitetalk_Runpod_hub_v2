"""zz_worker_vram: a ComfyUI custom node package that defines NO nodes.

Copied by Dockerfile.v24 to /ComfyUI/custom_nodes/zz_worker_vram. Records per-stage VRAM peaks
(sampler / VAE) for the handler's "[VRAM] vram peak ..." line, see stages.py. VRAM_STAGES=0
turns it off (nothing imported, nothing patched).
"""
import logging
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _enabled():
    return os.environ.get("VRAM_STAGES", "1").strip().lower() not in ("0", "false", "no", "off")


if _enabled():
    try:
        from . import stages
        stages.register()
    except Exception as e:
        logging.getLogger("worker_vram").warning(f"[VRAM] not enabled: {type(e).__name__}: {e}")
