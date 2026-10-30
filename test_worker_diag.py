import requests
import os
import time
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT_ID = "ghuoueuyp8iu7g"
API_KEY = os.environ["RUNPOD_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def run_diag(cmd):
    print(f"\n🔍 Running command on worker: '{cmd}'...", flush=True)
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    payload = {"input": {"cmd": cmd}}
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        res.raise_for_status()
        job_id = res.json()["id"]
        print(f"Job ID: {job_id}. Waiting for output...", flush=True)
        
        status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
        for _ in range(150):
            s_res = requests.get(status_url, headers=headers, timeout=15)
            if s_res.status_code == 200:
                data = s_res.json()
                if data.get("status") == "COMPLETED":
                    output = data.get("output", {})
                    print("\n📋 Command Output:", flush=True)
                    print(output.get("output", "No output field"), flush=True)
                    return output
                elif data.get("status") in ["FAILED", "CANCELLED"]:
                    print(f"❌ Job failed: {data}", flush=True)
                    return data
            time.sleep(2)
        print("⏱️ Timeout waiting for result.", flush=True)
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)

if __name__ == "__main__":
    test_cmd = sys.argv[1] if len(sys.argv) > 1 else "df -h && ls -la /runpod-volume"
    run_diag(test_cmd)
