# Deploy: one RunPod GPU pod (branch `deploy/pod`)

Same v32 image as serverless, **no rebuild**. The pod's start command runs `pod/start_pod.sh` from the network
volume, which copies this branch's worker code over the image, links the models from `/workspace/models`
(no copy) and starts an authenticated job API on port 8000. The backend (branch `deploy/pod`) talks to it with
`POD_BASE_URL` + `POD_API_TOKEN`. Going back to serverless: `DEPLOY_SERVERLESS.md` on `deploy/serverless`.

## 1. What runs on the pod

| Step | Where |
|---|---|
| `/runpod-volume -> /workspace` symlink (every volume path in the worker code works unchanged) | `start_pod.sh` |
| worker code overlay: `handler.py`, `setup_models.py`, `entrypoint.sh`, `worker_guard/*.py`, `pod/pod_api.py` | `start_pod.sh` |
| 8 models verified (names + exact bytes), never copied: `MODEL_SOURCE=baked`, `BAKED_MODELS_DIR=/workspace/models` | `check_models.sh`, `setup_models.py` |
| compile cache `/workspace/cache/compile` (kept across restarts) | `entrypoint.sh` |
| GPU guard, ComfyUI, startup check (same as serverless) | `entrypoint.sh` |
| job API on `0.0.0.0:8000`, Bearer token, one job at a time | `pod/pod_api.py` |

Why not `handler.py --rp_serve_api`: that is the RunPod SDK **development simulator**. Its `/run` only stores
the job, `POST /status` runs the whole 6+ min job inside that single request (longer than the RunPod proxy
allows) and then forgets it, and it has no auth. `pod_api.py` keeps the same routes and response shapes as
serverless (`/run`, `/status/{id}`, `/cancel/{id}`, `/health`) but runs jobs in a background thread and checks a
token. The simulator is still available for debugging with `POD_API=runpod_sim` (no auth: never expose it).

## 2. Pod template

| Setting | Value |
|---|---|
| Image | `digamberjhazaran/infinitetalk-runpod@sha256:423a8e0cd3a42ca4d26d82fffbae3292f76dcd0ba988a0280bfb703bc4970652` (v32) |
| GPU | 1x RTX 5090 (32 GB). Under 28 GB is rejected by the GPU guard |
| Container disk | 30 GB (models are not copied; temp files and ComfyUI outputs only) |
| Network volume | 60 GB+, mounted at `/workspace` (models 29.3 GB, +2.7 GB with Multi, compile cache, repo) |
| Expose HTTP ports | `8000` (job API). Optional TCP `22` for SSH |
| Start command | see below |

Start command (keeps the pod up in "setup mode" until the repo is on the volume):

```bash
bash -c 'S=/workspace/Infinitetalk_Runpod_hub_v2/pod/start_pod.sh; [ -f "$S" ] && exec bash "$S"; echo "[POD] setup mode: $S not found"; exec sleep infinity'
```

The image's ENTRYPOINT (`/opt/nvidia/nvidia_entrypoint.sh`) stays; the start command replaces only the CMD
(`/entrypoint.sh`), and `start_pod.sh` calls `/entrypoint.sh` itself at the end.

### Env vars

| Var | Value | Note |
|---|---|---|
| `POD_API_TOKEN` | `{{ RUNPOD_SECRET_pod_api_token }}` | **required**, 16+ chars. Create the secret in RunPod > Secrets. Same value as the backend's `POD_API_TOKEN` |
| `TORCH_COMPILE` | `1` | default |
| `FP8_FAST` | `0` | only after the R2/R3 quality test |
| `POD_MAX_QUEUE` | `10` | 429 above this |
| `POD_JOB_TTL_SEC` | `3600` | finished results kept this long (backend polls every 3.5 s) |
| `POD_REPO` / `POD_MODELS_DIR` | defaults `/workspace/Infinitetalk_Runpod_hub_v2`, `/workspace/models` | |
| `POD_ALLOW_MISSING_MODELS` | `0` | `1` starts even when check_models fails (debug) |
| `POD_API` | `pod` | `runpod_sim` = SDK simulator, no auth, debug only |

The public URL of the API is `https://<POD_ID>-8000.proxy.runpod.net` (backend `POD_BASE_URL`). It changes when
the pod is re-created (not when it is stopped and started).

## 3. One-time setup: getting the scripts and models onto /workspace

1. Create the network volume (same datacenter as the 5090s you want) and the pod from the template above.
   It starts in setup mode (`[POD] setup mode ...` in the log).
2. Open the pod's **Web Terminal** (or SSH) and put this branch on the volume. Either:
   - after the branch is pushed: `cd /workspace && git clone -b deploy/pod https://github.com/aiengineer-zarantech/Infinitetalk_Runpod_hub_v2.git`
     (private repo: use a fine-grained **read-only** GitHub token when git asks for the password), or
   - without a push: on the dev machine `git bundle create pod.bundle deploy/pod`, send it with
     `runpodctl send pod.bundle` and on the pod `runpodctl receive <code>` then
     `git clone -b deploy/pod pod.bundle /workspace/Infinitetalk_Runpod_hub_v2`.
3. Models (once, ~29 GB from huggingface.co, resumable):
   `bash /workspace/Infinitetalk_Runpod_hub_v2/pod/download_models.sh` (add `INCLUDE_MULTI=1` for multi-person).
   It ends with `[MODELS] all 8 required models OK in /workspace/models`.
   If the old serverless volume already holds the models, copying them into `/workspace/models/<folder>/` works
   too; `check_models.sh` verifies names and sizes.
4. Restart the pod. The log should show, in order: `[POD] /runpod-volume -> /workspace`, `[MODELS] all 8 ...`,
   `[TIMING] model setup ... 0.00 GB copied`, `✅ ComfyUI is ready!`, `=== Starting pod job API on 0.0.0.0:8000`.
   (setup_models labels the linked files `[HF_CACHE]`; with `MODEL_SOURCE=baked` they come from /workspace/models.)
5. Test from your machine:
   `curl -H "Authorization: Bearer <token>" https://<POD_ID>-8000.proxy.runpod.net/health`

Code update later: `cd /workspace/Infinitetalk_Runpod_hub_v2 && git pull` (or a new bundle), then restart the
pod: `start_pod.sh` copies the code over the image at every start.

## 4. Daily routine

Start:
1. RunPod console > Pods > Start (or `runpodctl start pod <POD_ID>`). If no 5090 is free on that host, wait or
   deploy a new pod on the same network volume (new POD_ID: update the backend's `POD_BASE_URL`).
2. Wait ~2-4 min (GPU guard, model links, ComfyUI). `/health` answers once the API is up; the Studio navbar
   turns green.
3. The first video after a start compiles (cache on `/workspace` makes it shorter than the very first time).

Stop:
1. Check nothing is running: `/health` shows `"running": 0` and `"inQueue": 0`. A restart or stop loses queued
   and running jobs (the queue is in memory); the Studio then shows "GPU server off hai".
2. RunPod console > Stop (or `runpodctl stop pod <POD_ID>`). The volume and the models stay; a stopped pod
   bills only storage.

A running pod bills every hour, idle or not.

## 5. Differences from serverless

| | serverless (`deploy/serverless`) | pod (`deploy/pod`) |
|---|---|---|
| Start | `/entrypoint.sh` -> `handler.py` (serverless loop) | `start_pod.sh` -> `/entrypoint.sh` -> `pod/pod_api.py` |
| Models | copied from `/runpod-volume/models` every cold start | linked from `/workspace/models`, no copy |
| Parallel chunks | on (up to 4 workers) | off (1 GPU, one job at a time) |
| API | api.runpod.ai `/v2/<endpoint>/...` with the RunPod key | `https://<POD_ID>-8000.proxy.runpod.net/...` with `POD_API_TOKEN` |
| Billing | per second of work + cold starts | per hour while the pod runs |
