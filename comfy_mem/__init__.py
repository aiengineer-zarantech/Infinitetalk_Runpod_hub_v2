"""zz_worker_mem: a ComfyUI custom node package that defines NO nodes (MODEL_LOCK helper).

POST /worker/clear_cache: gc.collect() + soft_empty_cache() + torch.cuda.ipc_collect() inside the ComfyUI
process. Frees memory that no tensor uses any more (allocator cache, dropped intermediates). It never unloads a
model and never touches ComfyUI's execution cache, unlike POST /free (free_memory also resets that cache, so the
next job would load the model again). Called by the handler only between jobs (ComfyUI idle).
Install: /ComfyUI/custom_nodes/zz_worker_mem (Dockerfile.v34 / pod/start_pod.sh).
"""
import gc
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
logger = logging.getLogger("worker_mem")


def _mem():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"free": int(free), "total": int(total), "reserved": int(torch.cuda.memory_reserved()),
                    "allocated": int(torch.cuda.memory_allocated())}
    except Exception:
        pass
    return {}


def clear():
    before = _mem()
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache(True)
    except Exception:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return {"before": before, "after": _mem()}


def register():
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/worker/clear_cache")
    async def worker_clear_cache(request):
        return web.json_response(clear())

    logger.info("[MEM] POST /worker/clear_cache (gc + empty_cache, models stay loaded)")


try:
    register()
except Exception as e:  # outside ComfyUI (tests) or an old server: the handler falls back to measuring only
    logger.warning(f"[MEM] not enabled: {type(e).__name__}: {e}")
