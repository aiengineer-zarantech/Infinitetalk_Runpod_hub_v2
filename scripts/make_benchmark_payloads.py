#!/usr/bin/env python3
# ==============================================================================
# make_benchmark_payloads.py - Writes the two A/B benchmark payloads (see TESTING.md):
#   bench_compile_off.json  {"input": {"benchmark": true, "torch_compile": false, ...}}
#   bench_compile_on.json   {"input": {"benchmark": true, "torch_compile": true,  ...}}
# Same image, audio, prompt and size in both; the worker trims the audio to 10 s and the
# workflow seed is fixed (2), so the only difference is torch.compile.
# Usage: python scripts\make_benchmark_payloads.py --image face.jpeg --audio voice.wav
# ==============================================================================
import argparse
import base64
import json
import os
import sys


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def payload(image_b64, audio_b64, args, torch_compile):
    job = {
        "benchmark": True,
        "torch_compile": torch_compile,
        "image_base64": image_b64,
        "wav_base64": audio_b64,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
    }
    if args.fp8_fast:
        job["fp8_fast"] = True
    return {"input": job}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True, help="WAV recommended (cut sample-exact); mp3/flac also work")
    parser.add_argument("--prompt", default="A person talking naturally")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fp8-fast", action="store_true", help="also set fp8_fast (changes numerics; test only)")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    image_b64, audio_b64 = b64(args.image), b64(args.audio)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, flag in (("bench_compile_off.json", False), ("bench_compile_on.json", True)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload(image_b64, audio_b64, args, flag), f)
        print(f"wrote {path} (torch_compile {str(flag).lower()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
