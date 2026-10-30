#!/usr/bin/env python3
"""
test_new_tts.py - Standalone test script for RunPod TTS Endpoint
"""

import os
import sys
import time
import json
import base64
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Default endpoint (can be overridden via CLI or ENV). The API key is never stored in this file:
# set RUNPOD_API_KEY in your own shell or pass --api-key.
DEFAULT_ENDPOINT_ID = "h2ow5pqidc8uqw"


def make_request(url, headers, data=None, timeout=30):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": raw}


def main():
    parser = argparse.ArgumentParser(description="Test New RunPod TTS Endpoint")
    parser.add_argument("--endpoint-id", default=os.getenv("TTS_ENDPOINT_ID", DEFAULT_ENDPOINT_ID), help="RunPod TTS Endpoint ID")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API Key (default: RUNPOD_API_KEY env var)")
    parser.add_argument("--text", default="Testing our new dedicated Chatterbox TTS endpoint on RunPod. Voice cloning is now active and ready.", help="Text to speak")
    parser.add_argument("--ref-audio", default="../2min25.mp3", help="Reference audio file (WAV or MP3)")
    parser.add_argument("--language", default="en", help="Language code (en, hi, etc.)")
    parser.add_argument("--output", default="./outputs/new_endpoint_test.wav", help="Output WAV path")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ Error: set RUNPOD_API_KEY in your shell or pass --api-key", file=sys.stderr)
        sys.exit(1)

    ref_path = Path(args.ref_audio)
    if not ref_path.is_file():
        # Fallback search
        alt_paths = [Path("../2min25.mp3"), Path("examples/audio.mp3"), Path("../infinitetalk_studio/backend/voices/en_placeholder_1.wav")]
        for p in alt_paths:
            if p.is_file():
                ref_path = p
                break

    if not ref_path.is_file():
        print(f"❌ Error: Reference audio file not found: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("🎙️ RUNPOD TTS ENDPOINT TEST")
    print("=" * 60)
    print(f"Endpoint ID : {args.endpoint_id[:4]}...")
    print(f"Ref Audio   : {ref_path} ({ref_path.stat().st_size / 1024:.1f} KB)")
    print(f"Text        : {args.text[:80]}...")
    print(f"Language    : {args.language}")
    print(f"Output Path : {args.output}")
    print("=" * 60)

    # Encode reference audio to base64
    audio_b64 = base64.b64encode(ref_path.read_bytes()).decode("utf-8")

    payload = {
        "input": {
            "task": "tts",
            "text": args.text,
            "language_id": args.language,
            "reference_audio_base64": audio_b64,
            "temperature": 0.8,
            "cfg_weight": 0.5,
            "exaggeration": 0.5,
            "seed": 42,
        }
    }

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    print("\n📡 Submitting job to /run...")
    run_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"
    status_code, res = make_request(run_url, headers, payload)
    if status_code != 200 or "id" not in res:
        print(f"❌ Submission failed (HTTP {status_code}): {res}", file=sys.stderr)
        sys.exit(1)

    job_id = res["id"]
    print(f"✅ Job submitted! Job ID: {job_id}")
    print("⏳ Waiting for generation (polling every 3s)...\n")

    status_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status/{job_id}"
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        s_code, s_res = make_request(status_url, headers)
        job_status = s_res.get("status", "UNKNOWN") if isinstance(s_res, dict) else "ERROR"
        print(f"⏱️ [{elapsed:5.1f}s] Status: {job_status}")

        if job_status == "COMPLETED":
            output = s_res.get("output", {})
            if isinstance(output, dict) and output.get("error"):
                print(f"\n❌ Worker reported failure: {output['error']}", file=sys.stderr)
                sys.exit(1)

            b64_out = output.get("audio_base64") if isinstance(output, dict) else None
            if not b64_out:
                print(f"\n❌ Worker returned COMPLETED but no audio_base64 in output: {output}", file=sys.stderr)
                sys.exit(1)

            if "," in b64_out:
                b64_out = b64_out.split(",", 1)[1]

            out_bytes = base64.b64decode(b64_out)
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(out_bytes)

            dur = output.get("duration_sec", 0.0) if isinstance(output, dict) else 0.0
            sr = output.get("sample_rate", 24000) if isinstance(output, dict) else 24000
            print("\n" + "=" * 60)
            print(f"🎉 [SUCCESS] Cloned voice generated in {elapsed:.1f}s!")
            print(f"💾 Saved to   : {out_file.resolve()}")
            print(f"📊 Duration   : {dur:.2f}s")
            print(f"🔊 Sample Rate: {sr} Hz ({len(out_bytes) / 1024:.1f} KB)")
            print("=" * 60)
            break

        if job_status in ("FAILED", "TIMED_OUT", "CANCELLED"):
            err = s_res.get("error") or s_res
            print(f"\n❌ Job failed with status {job_status}: {err}", file=sys.stderr)
            sys.exit(1)

        time.sleep(3.0)


if __name__ == "__main__":
    main()
