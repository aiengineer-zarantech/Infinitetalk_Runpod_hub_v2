#!/usr/bin/env python3
# ==============================================================================
# make_upscale_payloads.py - Writes the job JSONs for the v26 upscale tests (TESTING.md "v26"), in this order:
#
#   u1_upscale_only_3b.json   "mode": "upscale_only" on an existing 480p clip, SeedVR2 3B
#   u2_upscale_only_7b.json   same clip, SeedVR2 7B
#   u3_fast480_16x9.json      "mode": "fast480" (848x480 -> 1280x720), upscale 3B (the default)
#   u4_fast480_9x16.json      "mode": "fast480", "aspect": "9:16" (480x848 -> 720x1280)
#
# The files contain the video / image / audio as base64: they are git-ignored, rebuild them with this script.
# RunPod rejects /run bodies above 10 MB: keep the clip short (10 s of 480p is about 1-3 MB) or use --video-url.
# Usage: python scripts\make_upscale_payloads.py --video clip480.mp4 [--seconds 10] --image face.jpeg --audio voice.wav
#        python scripts\make_upscale_payloads.py --video-url https://.../clip480.mp4 --from payloads\phaseA\a2_fast480_16x9.json
# ==============================================================================
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

PROMPT = "A person talking naturally"
MAX_BODY_MB = 9.5


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def cut(video, seconds):
    """First <seconds> of the clip: video re-encoded like the generation output (crf 19), audio copied."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("--seconds needs ffmpeg on PATH (or cut the clip yourself and leave --seconds out)")
    out = os.path.join(tempfile.mkdtemp(), "clip.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-t", str(seconds), "-c:v", "libx264", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", out], check=True)
    return out


def generation_media(args):
    if args.source:
        with open(args.source, encoding="utf-8-sig") as f:
            src = json.load(f)["input"]
        return src["image_base64"], src["wav_base64"], src.get("prompt", args.prompt)
    if args.image and args.audio:
        return b64(args.image), b64(args.audio), args.prompt
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="existing 480p clip (848x480 or 480x848) for the upscale_only jobs")
    parser.add_argument("--video-url", help="public URL of the clip instead of --video (no size limit)")
    parser.add_argument("--seconds", type=float, help="use only the first N seconds of --video (needs ffmpeg)")
    parser.add_argument("--from", dest="source", help="existing payload to take image_base64 / wav_base64 from")
    parser.add_argument("--image")
    parser.add_argument("--audio", help="15-20 s voice for the full fast480 jobs")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default=os.path.join("payloads", "v26"))
    args = parser.parse_args(argv)

    jobs = {}
    if args.video or args.video_url:
        if args.video_url:
            video = {"video_url": args.video_url}
        else:
            video = {"video_base64": b64(cut(args.video, args.seconds) if args.seconds else args.video)}
        jobs["u1_upscale_only_3b.json"] = {"input": {"mode": "upscale_only", "upscale_model": "3b", **video}}
        jobs["u2_upscale_only_7b.json"] = {"input": {"mode": "upscale_only", "upscale_model": "7b", **video}}
    media = generation_media(args)
    if media:
        image_b64, audio_b64, prompt = media
        base = {"image_base64": image_b64, "wav_base64": audio_b64, "prompt": prompt, "mode": "fast480"}
        jobs["u3_fast480_16x9.json"] = {"input": {**base}}
        jobs["u4_fast480_9x16.json"] = {"input": {**base, "aspect": "9:16"}}
    if not jobs:
        raise SystemExit("give --video / --video-url (upscale_only jobs) and/or --from or --image + --audio (fast480 jobs)")

    os.makedirs(args.out, exist_ok=True)
    for name, payload in jobs.items():
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        size_mb = os.path.getsize(path) / 1024 ** 2
        shown = {k: v for k, v in payload["input"].items() if not k.endswith("_base64")}
        warn = f"  WARNING: {size_mb:.1f} MB, above RunPod's /run limit; use --video-url" if size_mb > MAX_BODY_MB else ""
        print(f"{path} ({size_mb:.1f} MB): {json.dumps(shown)}{warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
