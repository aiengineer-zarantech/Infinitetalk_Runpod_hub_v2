#!/usr/bin/env python3
"""
test_tts_endpoint.py - RunPod Serverless Chatterbox Voice Cloning CLI Test Script

Tests the Chatterbox TTS endpoint on RunPod by providing a text prompt and
a reference audio clip (local WAV/MP3 file or URL).
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
import functools
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print = functools.partial(print, flush=True)


def get_runpod_api_key():
    """Resolve RunPod API key from environment without printing or logging it."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip().strip('"').strip("'")

    candidates = [
        Path("../infinitetalk_studio/backend/.env"),
        Path("test_worker_diag.py"),
        Path(".env"),
        Path("../.env"),
    ]
    for c in candidates:
        if c.exists():
            try:
                for line in c.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line_s = line.strip()
                    if line_s.startswith("#"):
                        continue
                    if "RUNPOD_API_KEY" in line_s and "=" in line_s:
                        val = line_s.split("=", 1)[1].strip()
                        if '"' in val:
                            extracted = val.split('"')[1]
                        elif "'" in val:
                            extracted = val.split("'")[1]
                        else:
                            extracted = val.split()[0]
                        if extracted and extracted.startswith("rpa_"):
                            return extracted
            except Exception:
                pass
    return None


def get_tts_endpoint_id():
    """Resolve TTS Endpoint ID from environment or configuration."""
    ep = os.environ.get("TTS_ENDPOINT_ID")
    if ep and ep.strip():
        return ep.strip().strip('"').strip("'")

    candidates = [
        Path("../infinitetalk_studio/backend/.env"),
        Path(".env"),
        Path("../.env"),
    ]
    for c in candidates:
        if c.exists():
            try:
                for line in c.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line_s = line.strip()
                    if line_s.startswith("#"):
                        continue
                    if "TTS_ENDPOINT_ID" in line_s and "=" in line_s:
                        val = line_s.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
            except Exception:
                pass
    return None


def make_request(url, headers, data=None, timeout=30):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": raw}
    except Exception as e:
        return 0, {"error": str(e)}


def load_reference_audio(audio_source: str) -> bytes:
    """Loads audio bytes from either a local file or an HTTP/HTTPS URL."""
    if audio_source.startswith("http://") or audio_source.startswith("https://"):
        print(f"📥 Downloading reference audio from: {audio_source}...")
        req = urllib.request.Request(audio_source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        print(f"✅ Downloaded {len(data) / 1024:.1f} KB")
        return data
    else:
        path = Path(audio_source)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio file not found: {path}")
        return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description="Test RunPod Chatterbox Voice Cloning Endpoint")
    parser.add_argument("--endpoint-id", type=str, default=None, help="RunPod TTS Endpoint ID")
    parser.add_argument("--api-key", type=str, default=None, help="RunPod API Key")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the Chatterbox voice cloning system on RunPod. Looking realistic does not mean behaving realistically.",
        help="Text prompt to speak (no length limit; worker splits into sentence chunks)",
    )
    parser.add_argument("--text-file", type=str, default=None, help="Optional text file containing prompt")
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path or URL to reference audio WAV/MP3 (5-15 seconds recommended)",
    )
    parser.add_argument("--language-id", type=str, default="en", help="Language code (e.g. en, hi, es, fr, etc.)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (0.05 - 2.0)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight (0.0 - 1.0)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration (0.25 - 2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for generation")
    parser.add_argument("--output-path", type=str, default="./outputs/cloned_voice.wav", help="Path to save output WAV")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and payload without submitting to RunPod")

    args = parser.parse_args()

    # 1. Resolve Text
    text = args.text
    if args.text_file:
        tf = Path(args.text_file)
        if not tf.is_file():
            print(f"❌ Error: text-file not found: {tf}", file=sys.stderr)
            sys.exit(1)
        text = tf.read_text(encoding="utf-8").strip()

    if not text:
        print("❌ Error: Text prompt cannot be empty.", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve Reference Audio
    ref_audio_arg = args.ref_audio
    if not ref_audio_arg:
        # Fallback to local sample presets if available
        candidates = [
            Path("../infinitetalk_studio/backend/voices/en_placeholder_1.wav"),
            Path("examples/audio.mp3"),
            Path("../2min25.mp3"),
        ]
        for c in candidates:
            if c.is_file():
                ref_audio_arg = str(c)
                break

    if not ref_audio_arg:
        print("❌ Error: Reference audio not specified and no default sample audio found.", file=sys.stderr)
        print("Provide --ref-audio <path-or-url>", file=sys.stderr)
        sys.exit(1)

    try:
        audio_bytes = load_reference_audio(ref_audio_arg)
    except Exception as e:
        print(f"❌ Error loading reference audio: {e}", file=sys.stderr)
        sys.exit(1)

    if len(audio_bytes) < 100:
        print(f"❌ Error: Reference audio is too small ({len(audio_bytes)} bytes).", file=sys.stderr)
        sys.exit(1)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # 3. Resolve Endpoint & API Key
    endpoint_id = args.endpoint_id or get_tts_endpoint_id()
    if not endpoint_id:
        print("❌ Error: TTS_ENDPOINT_ID could not be found. Provide --endpoint-id or set TTS_ENDPOINT_ID.", file=sys.stderr)
        sys.exit(1)

    api_key = (args.api_key or "").strip() or get_runpod_api_key()
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY could not be found in environment or config.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("🎙️ [TEST_TTS_ENDPOINT] Chatterbox Voice Cloning Request")
    print("=" * 60)
    print(f"Endpoint ID     : {endpoint_id[:4]}...")
    print(f"Language ID     : {args.language_id}")
    print(f"Reference Audio : {ref_audio_arg} ({len(audio_bytes)/1024:.1f} KB)")
    print(f"Text Length     : {len(text)} characters ({len(text.split())} words)")
    print(f"Text Preview    : {text[:80]}...")
    print(f"Temperature     : {args.temperature}")
    print(f"CFG Weight      : {args.cfg_weight}")
    print(f"Exaggeration    : {args.exaggeration}")
    print(f"Seed            : {args.seed}")
    print(f"Output File     : {args.output_path}")
    print("=" * 60)

    payload = {
        "input": {
            "task": "tts",
            "text": text,
            "language_id": args.language_id,
            "reference_audio_base64": audio_b64,
            "temperature": args.temperature,
            "cfg_weight": args.cfg_weight,
            "exaggeration": args.exaggeration,
            "seed": args.seed,
        }
    }

    if args.dry_run:
        print("🔍 [DRY-RUN] Validation passed. Payload prepared successfully without submitting.")
        sys.exit(0)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    print("\n📡 Submitting job to /run...")
    code, res = make_request(run_url, headers, payload)
    if code != 200 or "id" not in res:
        print(f"❌ Failed to submit job! Status {code}: {res}", file=sys.stderr)
        sys.exit(1)

    job_id = res["id"]
    print(f"✅ Job submitted successfully! Job ID: {job_id}")
    print("⏳ Polling /status every 3s...\n")

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        code, status_data = make_request(status_url, headers)

        if code == 404:
            print(f"⏱️ [{elapsed:5.1f}s] Status: INITIALIZING / PENDING")
            time.sleep(3)
            continue
        elif code != 200:
            print(f"⏱️ [{elapsed:5.1f}s] Polling error ({code}): {status_data.get('error', 'unknown')}")
            time.sleep(3)
            continue

        current_status = status_data.get("status")
        print(f"⏱️ [{elapsed:5.1f}s] Status: {current_status}")

        output = status_data.get("output") or {}
        has_error = (current_status in ["FAILED", "CANCELLED", "TIMED_OUT"]) or ("error" in output)

        if has_error:
            print("\n" + "=" * 60)
            print(f"❌ TTS GENERATION FAILED (Job ID: {job_id})")
            print("=" * 60)
            print(output.get("error") or status_data.get("error") or status_data)
            print("=" * 60)
            sys.exit(1)

        elif current_status == "COMPLETED":
            audio_b64 = output.get("audio_base64")
            if not audio_b64:
                print(f"⚠️ Job COMPLETED but no audio_base64 returned: {output}", file=sys.stderr)
                sys.exit(1)

            if "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]

            wav_data = base64.b64decode(audio_b64)
            out_file = Path(args.output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(wav_data)

            duration = output.get("duration_sec", 0.0)
            sr = output.get("sample_rate", 24000)
            chunks = output.get("chunks", 1)

            print("\n" + "=" * 60)
            print(f"🎉 [SUCCESS] Cloned voice generated in {elapsed:.1f}s!")
            print(f"💾 Saved to   : {out_file.resolve()}")
            print(f"📊 Duration   : {duration:.2f}s ({chunks} chunk{'s' if chunks != 1 else ''})")
            print(f"🔊 Sample Rate: {sr} Hz")
            print(f"📦 File Size  : {len(wav_data) / 1024:.1f} KB")
            print("=" * 60)
            break

        time.sleep(3)


if __name__ == "__main__":
    main()
