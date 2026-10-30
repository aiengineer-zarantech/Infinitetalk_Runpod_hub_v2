#!/usr/bin/env python3
"""Runs once ComfyUI is ready: the device must be cuda, then prints the [STARTUP] timing line.

Timestamps are epoch seconds from `date +%s.%N` in entrypoint.sh. Exit code 1 stops the worker.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker_guard import comfy_api  # noqa: E402
from worker_guard.job_guard import DEFAULT_VRAM_NEAR_ZERO_MB  # noqa: E402

STATS_ATTEMPTS = 5


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=float, required=True, help="startup began")
    parser.add_argument("--gpu-done", type=float, required=True, help="GPU guard finished, copy + ComfyUI started")
    parser.add_argument("--copy-done", type=float, required=True, help="model copy finished")
    parser.add_argument("--comfy-ready", type=float, required=True, help="ComfyUI answered HTTP")
    args = parser.parse_args(argv)

    device, last_error = None, None
    for attempt in range(STATS_ATTEMPTS):
        try:
            device = comfy_api.first_device(comfy_api.system_stats(timeout=10))
            break
        except Exception as e:
            last_error = e
            if attempt < STATS_ATTEMPTS - 1:
                time.sleep(2)
    else:
        print(f"[COMFY] FAILED: /system_stats unreachable ({type(last_error).__name__}: {last_error})", flush=True)
        return 1
    if device is None:
        print("[COMFY] FAILED: /system_stats reported no devices", flush=True)
        return 1

    gb = 1024 ** 3
    print(
        f"[COMFY] device {device['type']}, {device['name']}, "
        f"VRAM used {device['vram_used'] / gb:.1f} / {device['vram_total'] / gb:.1f} GB",
        flush=True,
    )
    if device["type"] != "cuda":
        print(f"[COMFY] FAILED: device type is '{device['type']}', expected cuda", flush=True)
        return 1

    now = time.time()
    print(
        f"[STARTUP] ready in {now - args.t0:.1f}s (gpu_check {args.gpu_done - args.t0:.1f}s, "
        f"copy {args.copy_done - args.gpu_done:.1f}s, comfy {args.comfy_ready - args.gpu_done:.1f}s)",
        flush=True,
    )
    threshold = os.environ.get("VRAM_NEAR_ZERO_MB", str(DEFAULT_VRAM_NEAR_ZERO_MB))
    print(
        f"[STARTUP] idle VRAM used {device['vram_used'] / (1024 ** 2):.0f} MB "
        f"(watchdog near-zero threshold VRAM_NEAR_ZERO_MB={threshold})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
