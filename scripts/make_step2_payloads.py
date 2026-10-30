#!/usr/bin/env python3
# ==============================================================================
# make_step2_payloads.py - Writes the four 720p OOM diagnosis payloads (TESTING.md step 2):
#   step2a_object_info.json        cmd job: wrapper node inputs + where tiled_vae / vae.decode are used
#   step2b_repro_720p.json         1280x720, benchmark (10 s audio), manual 0 / False, compile off
#   step2c_vae_tiling_720p.json    2b + "vae_tiling": true   (only if 2b fails in the VAE)
#   step2d_measure_25_blocks.json  2b with blocks_to_swap 25 (only if 2b fails in the transformer)
# 2b-2d need OFFLOAD_ALLOW_MANUAL=1 and PROFILE=1 on the endpoint.
# Usage: python scripts\make_step2_payloads.py --from ..\.baselines\bench\bench_compile_off.json
#        python scripts\make_step2_payloads.py --image face.jpeg --audio voice.wav
# ==============================================================================
import argparse
import base64
import json
import os
import sys

OBJECT_INFO_NODES = ("WanVideoImageToVideoMultiTalk", "WanVideoDecode", "WanVideoEncode", "WanVideoVAELoader",
                     "WanVideoSampler")
OBJECT_INFO_PY = (
    "import json, urllib.request\n"
    f"for n in {OBJECT_INFO_NODES!r}:\n"
    "    d = json.load(urllib.request.urlopen('http://127.0.0.1:8188/object_info/' + n, timeout=30)).get(n, {})\n"
    "    print('== ' + n + (' (NOT INSTALLED)' if not d else ''))\n"
    "    for kind in ('required', 'optional'):\n"
    "        for k, v in d.get('input', {}).get(kind, {}).items():\n"
    "            print('   ', kind, k, json.dumps(v)[:200])\n"
)
WRAPPER = "/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"
CMD = (
    f"cd {WRAPPER} && echo '== wrapper commit' && git log -1 --format='%h %ci %s' 2>&1; "
    f"python3 -c \"{OBJECT_INFO_PY}\" 2>&1; "
    "echo '== tiled_vae in wrapper source'; grep -rn --include=*.py 'tiled_vae' . | head -60; "
    "echo '== vae.decode / vae.encode calls'; grep -rn --include=*.py -E 'vae\\.(decode|encode)\\(' . | head -60"
)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def media(args):
    if args.source:
        with open(args.source, encoding="utf-8") as f:
            src = json.load(f)["input"]
        return src["image_base64"], src["wav_base64"], src.get("prompt", args.prompt)
    if not (args.image and args.audio):
        raise SystemExit("give --from <bench payload> or both --image and --audio")
    return b64(args.image), b64(args.audio), args.prompt


def repro(image_b64, audio_b64, prompt, **extra):
    job = {
        "benchmark": True,          # audio cut to the first 10 s (4 windows)
        "torch_compile": False,     # honoured for benchmark jobs only; keeps compile memory out of the picture
        "offload_mode": "manual",   # needs OFFLOAD_ALLOW_MANUAL=1; manual jobs get no emergency OOM retry
        "blocks_to_swap": 0,
        "force_offload": False,
        "width": 1280,
        "height": 720,
        "prompt": prompt,
        "image_base64": image_b64,
        "wav_base64": audio_b64,
    }
    job.update(extra)
    return {"input": job}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", help="reuse image/audio/prompt from an existing payload JSON")
    parser.add_argument("--image")
    parser.add_argument("--audio", help="WAV recommended (benchmark trims it sample-exact)")
    parser.add_argument("--prompt", default="A person talking naturally")
    parser.add_argument("--out-dir", default=os.path.join("payloads", "step2"))
    args = parser.parse_args(argv)

    image_b64, audio_b64, prompt = media(args)
    payloads = {
        "step2a_object_info.json": {"input": {"cmd": CMD}},
        "step2b_repro_720p.json": repro(image_b64, audio_b64, prompt),
        "step2c_vae_tiling_720p.json": repro(image_b64, audio_b64, prompt, vae_tiling=True),
        "step2d_measure_25_blocks.json": repro(image_b64, audio_b64, prompt, blocks_to_swap=25),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, payload in payloads.items():
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
