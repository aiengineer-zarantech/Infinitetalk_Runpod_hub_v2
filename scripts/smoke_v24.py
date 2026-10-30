#!/usr/bin/env python3
# ==============================================================================
# smoke_v24.py - Build-time check (Dockerfile.v24) on the real workflow files, CPU only, writes nothing:
# fast480 queues blocks_to_swap 0 / force_offload False / main_device on a 31.4 GB GPU, 720p keeps the
# tier values (25 / True) while not calibrated, the OOM ladder ends with VAE tiling after 40, and a job
# without "mode" is returned unchanged. Usage (from /): python /tmp/smoke_v24.py [workflow dir]
# ==============================================================================
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from worker_guard import modes, offload  # noqa: E402

GB = 1024 ** 3


def workflow(path, width, height):
    with open(path, encoding="utf-8") as f:
        prompt = json.load(f)
    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height
    return prompt


def main(workflow_dir="/"):
    single = os.path.join(workflow_dir, "I2V_single.json")
    for aspect, size in (("16:9", (848, 480)), ("9:16", (480, 848))):
        job, error = modes.prepare_job_input({"mode": "fast480", "aspect": aspect, "width": 1280, "height": 720})
        assert error is None and (job["width"], job["height"]) == size, (aspect, job, error)
        decision, _ = offload.decide(job, 31.4 * GB, vram_used_bytes=GB)
        assert (decision.blocks_to_swap, decision.force_offload) == (0, False), decision
        prompt = workflow(single, *size)
        assert prompt["134"]["inputs"]["blocks_to_swap"] == 38  # workflow default, fixed by the final check
        assert offload.enforce_final(prompt, 31.4, (0, False)) == (0, False)
        assert prompt["122"]["inputs"]["load_device"] == "main_device"
    decision, _ = offload.decide({"width": 1280, "height": 720}, 31.4 * GB, vram_used_bytes=GB)
    assert (decision.blocks_to_swap, decision.force_offload) == (25, True) and not decision.prediction.calibrated
    assert offload.oom_ladder(0) == [(10, False), (25, False), (40, False), (40, True)]
    assert offload.oom_ladder(25)[-2:] == [(40, False), (40, True)]
    old = {"width": 720, "height": 480}
    assert modes.prepare_job_input(old) == (old, None)
    print("OK: fast480 / adaptive / ladder smoke check")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
