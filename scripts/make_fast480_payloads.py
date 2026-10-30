#!/usr/bin/env python3
# ==============================================================================
# make_fast480_payloads.py - Writes the job JSONs for the fast480 phases (TESTING.md "Phase A").
# Same image, audio, prompt and seed (fixed in the workflow) in every job.
#
# Phase A (payloads\phaseA\):
#   a2_normal_480p.json            854x480, standard mode            -> "[OFFLOAD] final: 0 / False / main_device"
#   a2_fast480_16x9.json           "mode": "fast480" (848x480)       -> same, run it twice for A.3 (resident)
#   a2_fast480_9x16.json           fast480, "aspect": "9:16" (480x848)
#   a4_720p_b{0,5,10}_tiling_{off,on}.json
#                                  1280x720, manual blocks_to_swap 0/5/10, force_offload False, VAE tiling
#                                  off/on (needs OFFLOAD_ALLOW_MANUAL=1); calibration data for the adaptive offload
#   a5_ladder_720p.json            1280x720 in auto mode: with the OFFLOAD_TIERS env from TESTING.md it starts at
#                                  0 blocks, runs out of memory and shows the OOM ladder
# Every job is a benchmark job: the worker cuts the audio to the first 10 s (4 windows).
# The files contain the image and audio as base64: they are git-ignored, rebuild them with this script.
# Usage: python scripts\make_fast480_payloads.py --from ..\.baselines\bench\bench_compile_off.json
#        python scripts\make_fast480_payloads.py --image face.jpeg --audio voice.wav
# ==============================================================================
import argparse
import base64
import json
import os
import sys

PROMPT = "A person talking naturally"


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def media(args):
    if args.source:
        with open(args.source, encoding="utf-8-sig") as f:
            src = json.load(f)["input"]
        return src["image_base64"], src["wav_base64"], src.get("prompt", args.prompt)
    if not (args.image and args.audio):
        raise SystemExit("give --from <bench payload> or both --image and --audio")
    return b64(args.image), b64(args.audio), args.prompt


def phase_a(image_b64, audio_b64, prompt):
    def job(**extra):
        return {"input": {"benchmark": True, "image_base64": image_b64, "wav_base64": audio_b64,
                          "prompt": prompt, **extra}}

    jobs = {
        "a2_normal_480p.json": job(width=854, height=480),
        "a2_fast480_16x9.json": job(mode="fast480"),
        "a2_fast480_9x16.json": job(mode="fast480", aspect="9:16"),
        "a5_ladder_720p.json": job(width=1280, height=720),
    }
    for blocks in (0, 5, 10):
        for tiling in (False, True):
            name = f"a4_720p_b{blocks}_tiling_{'on' if tiling else 'off'}.json"
            jobs[name] = job(width=1280, height=720, offload_mode="manual", blocks_to_swap=blocks,
                             force_offload=False, vae_tiling=tiling)
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", help="existing payload to take image_base64 / wav_base64 from")
    parser.add_argument("--image")
    parser.add_argument("--audio", help="WAV recommended (cut sample-exact); mp3/flac also work")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default=os.path.join("payloads", "phaseA"))
    args = parser.parse_args(argv)

    image_b64, audio_b64, prompt = media(args)
    os.makedirs(args.out, exist_ok=True)
    jobs = phase_a(image_b64, audio_b64, prompt)
    for name, payload in jobs.items():
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        shown = {k: v for k, v in payload["input"].items() if not k.endswith("_base64")}
        print(f"{os.path.join(args.out, name)}: {json.dumps(shown)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
