#!/usr/bin/env python3
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
print = functools.partial(print, flush=True)

def get_runpod_api_key():
    """Resolve RunPod API key from environment without printing or logging it."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip().strip('"').strip("'")

    candidates = [
        Path("test_worker_diag.py"),
        Path(".env"),
        Path("../.env"),
    ]
    for c in candidates:
        if c.exists():
            try:
                for line in c.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "API_KEY" in line and "=" in line:
                        val = line.split("=", 1)[1].strip()
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

def run_diagnostic_cmd(endpoint_id, api_key, cmd):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    status_code, res = make_request(url, headers, {"input": {"cmd": cmd}})
    if status_code != 200 or "id" not in res:
        return f"Failed to submit diagnostic job: {res}"
    
    diag_job_id = res["id"]
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{diag_job_id}"
    for _ in range(60):
        time.sleep(3)
        code, sdata = make_request(status_url, headers)
        if code == 200:
            status = sdata.get("status")
            if status == "COMPLETED":
                return sdata.get("output", {}).get("output", "No output returned")
            elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                return f"Diagnostic job ended with status {status}: {sdata}"
    return "Diagnostic job timed out waiting for response."

def main():
    parser = argparse.ArgumentParser(description="Test RunPod InfiniteTalk Endpoint")
    parser.add_argument("--endpoint-id", type=str, default="ghuoueuyp8iu7g", help="RunPod Endpoint ID")
    parser.add_argument("--blocks-to-swap", type=int, default=0, help="Number of transformer blocks to swap to CPU")
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://raw.githubusercontent.com/aiengineer-zarantech/datasets/main/Gemini_Generated_Image_atuku9atuku9atuk.png",
        help="Input image URL"
    )
    parser.add_argument(
        "--wav-url",
        type=str,
        default="https://raw.githubusercontent.com/thevasudevgupta/gsoc-wav2vec2/main/data/sample.wav",
        help="Input audio WAV URL (3-5 seconds)"
    )
    parser.add_argument("--person-count", type=str, default="single", choices=["single", "multi"], help="Person count mode")
    parser.add_argument("--input-type", type=str, default="image", choices=["image", "video"], help="Input media type")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--outputs-dir", type=str, default="./outputs", help="Directory to save generated videos")
    parser.add_argument("--logs-dir", type=str, default="./logs", help="Directory to save logs and diagnostics")
    parser.add_argument("--always-diag", action="store_true", help="Always run diagnostic job even on success")

    args = parser.parse_args()

    api_key = get_runpod_api_key()
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY could not be found in environment or configuration files.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("🚀 [TEST_ENDPOINT] Starting RunPod InfiniteTalk Test")
    print("=" * 60)
    print(f"Endpoint ID    : {args.endpoint_id}")
    print(f"Input Type     : {args.input_type}")
    print(f"Person Count   : {args.person_count}")
    print(f"Resolution     : {args.width}x{args.height}")
    print(f"Blocks to Swap : {args.blocks_to_swap}")
    print(f"Image URL      : {args.image_url}")
    print(f"Audio WAV URL  : {args.wav_url}")
    print("=" * 60)

    payload = {
        "input": {
            "input_type": args.input_type,
            "person_count": args.person_count,
            "image_url": args.image_url,
            "wav_url": args.wav_url,
            "width": args.width,
            "height": args.height,
            "blocks_to_swap": args.blocks_to_swap,
            "prompt": "A person is talking naturally.",
            "network_volume": False
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    run_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"
    print("\n📡 Submitting job to /run...")
    code, res = make_request(run_url, headers, payload)
    if code != 200 or "id" not in res:
        print(f"❌ Failed to submit job! Status {code}: {res}", file=sys.stderr)
        sys.exit(1)

    job_id = res["id"]
    print(f"✅ Job submitted successfully! Job ID: {job_id}")
    print("⏳ Polling /status every 5s...\n")

    status_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status/{job_id}"
    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time
        code, status_data = make_request(status_url, headers)

        if code == 404:
            print(f"⏱️ [{elapsed:6.1f}s] Status: INITIALIZING / PENDING")
            time.sleep(5)
            continue
        elif code != 200:
            print(f"⏱️ [{elapsed:6.1f}s] Polling error ({code}): {status_data.get('error', 'unknown')}")
            time.sleep(5)
            continue

        current_status = status_data.get("status")
        print(f"⏱️ [{elapsed:6.1f}s] Status: {current_status}")

        output = status_data.get("output") or {}
        has_error = (current_status in ["FAILED", "CANCELLED", "TIMED_OUT"]) or ("error" in output)

        if has_error:
            os.makedirs(args.logs_dir, exist_ok=True)
            log_path = os.path.join(args.logs_dir, f"{job_id}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2)

            print("\n" + "=" * 60)
            print(f"❌ JOB FAILED or RETURNED ERROR (Job ID: {job_id})")
            print("=" * 60)
            print(f"📄 Full response saved to: {log_path}\n")

            err_val = output.get("error") or status_data.get("error")
            print("--- [error] ---")
            print(err_val)

            if "details" in output:
                print("\n--- [details] ---")
                print(json.dumps(output["details"], indent=2))

            if "comfyui_log" in output:
                print("\n--- [comfyui_log] ---")
                print(output["comfyui_log"])

            if "memory_events" in output:
                print("\n--- [memory_events] ---")
                print(output["memory_events"])

            if "gpu_state" in output:
                print("\n--- [gpu_state] ---")
                print(output["gpu_state"])

            if "comfyui_pid" in output:
                print("\n--- [comfyui_pid] ---")
                print(output["comfyui_pid"])
            print("=" * 60)

            # Follow-up diagnostic command
            print("\n🔧 Sending immediate follow-up diagnostic job to worker...")
            diag_cmd = "tail -n 200 /ComfyUI/comfyui.log; echo ---; cat /sys/fs/cgroup/memory.events; echo ---; nvidia-smi; echo ---; df -h /"
            diag_out = run_diagnostic_cmd(args.endpoint_id, api_key, diag_cmd)

            diag_path = os.path.join(args.logs_dir, f"{job_id}_diag.txt")
            with open(diag_path, "w", encoding="utf-8") as df:
                df.write(diag_out)

            print(f"📄 Diagnostic output saved to: {diag_path}\n")
            print("=" * 60)
            print("🔍 DIAGNOSTIC WORKER OUTPUT:")
            print("=" * 60)
            print(diag_out)
            print("=" * 60)
            sys.exit(1)

        elif current_status == "COMPLETED":
            os.makedirs(args.outputs_dir, exist_ok=True)
            video_b64 = output.get("video")
            if video_b64:
                if "," in video_b64:
                    video_b64 = video_b64.split(",", 1)[1]
                video_bytes = base64.b64decode(video_b64)
                out_path = os.path.join(args.outputs_dir, f"{job_id}.mp4")
                with open(out_path, "wb") as vf:
                    vf.write(video_bytes)
                print("\n" + "=" * 60)
                print(f"🎉 [SUCCESS] Video generated successfully in {elapsed:.1f}s!")
                print(f"💾 Saved to: {out_path} ({len(video_bytes)/(1024*1024):.2f} MB)")
                print("=" * 60)
            elif "video_path" in output:
                print("\n" + "=" * 60)
                print(f"🎉 [SUCCESS] Video saved to network volume: {output['video_path']}")
                print("=" * 60)
            else:
                print(f"\n⚠️ Job COMPLETED but no video found in output: {output}")

            if args.always_diag:
                print("\n🔧 Running diagnostic job to inspect timings and container state...")
                diag_cmd = "tail -n 200 /ComfyUI/comfyui.log; echo ---; cat /sys/fs/cgroup/memory.events; echo ---; nvidia-smi; echo ---; df -h /"
                diag_out = run_diagnostic_cmd(args.endpoint_id, api_key, diag_cmd)
                diag_path = os.path.join(args.logs_dir, f"{job_id}_diag.txt")
                os.makedirs(args.logs_dir, exist_ok=True)
                with open(diag_path, "w", encoding="utf-8") as df:
                    df.write(diag_out)
                print(f"📄 Diagnostic output saved to: {diag_path}")
                print(diag_out)
            break

        time.sleep(5)

if __name__ == "__main__":
    main()
