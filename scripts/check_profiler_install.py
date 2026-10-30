# ==============================================================================
# check_profiler_install.py - Build-time check (Dockerfile.v22), CPU only, no server, no network:
# loads the INSTALLED ComfyUI-WanVideoWrapper the way ComfyUI does, installs
# zz_worker_profiler on the loaded modules, and times one tiny real VAE encode.
# Exit 1 if the profiler would not install on this image.
# Usage (from anywhere): python check_profiler_install.py [comfy_dir]
# ==============================================================================
import asyncio
import importlib.util
import os
import sys
import types


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    comfy_dir = argv[0] if argv else "/ComfyUI"
    sys.argv = ["main.py", "--cpu"]
    os.chdir(comfy_dir)
    sys.path.insert(0, comfy_dir)

    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import server
    # The wrapper's latent_preview reads PromptServer.instance at import time
    server.PromptServer.instance = types.SimpleNamespace(
        add_on_prompt_handler=lambda fn: None, send_sync=lambda *a, **k: None, routes=None, client_id=None)

    wrapper = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-WanVideoWrapper")
    if not asyncio.run(nodes.load_custom_node(wrapper, module_parent="custom_nodes")):
        print("PROFILER INSTALL CHECK FAILED: the wrapper did not load")
        return 1

    path = os.path.join(comfy_dir, "custom_nodes", "zz_worker_profiler", "profiler.py")
    spec = importlib.util.spec_from_file_location("zz_worker_profiler_check.profiler", path)
    profiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler)
    if profiler.install() is not True:
        print("PROFILER INSTALL CHECK FAILED: install() returned False (see the [PROFILE] warning above)")
        return 1

    import torch
    vae_mod = next(m for m in list(sys.modules.values())
                   if profiler._norm(getattr(m, "__file__", None)).endswith("/ComfyUI-WanVideoWrapper/wanvideo/wan_video_vae.py"))
    vae = vae_mod.WanVideoVAE(dtype=torch.float32)
    profiler._active = rec = profiler.LoopRecorder(profiler._clock())
    try:
        vae.encode(torch.zeros(1, 3, 5, 16, 16), device="cpu", pbar=False)
    finally:
        profiler._active = None
    events = [(e[0], e[3]) for e in rec.current.events]
    if events != [("vae_encode", 5)]:
        print(f"PROFILER INSTALL CHECK FAILED: unexpected events {events}")
        return 1
    print("profiler install check OK: real wrapper modules patched, VAE encode timed (5 frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
