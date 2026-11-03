# Deploy: RunPod Serverless (branch `deploy/serverless`)

Branch layout (worker and backend repos use the same names):

```
perf/5090-speedup        common fixes for both modes (setup_models modes + resilient copy, fp8 fix,
   |                     timing logs, vocal skip, SeedVR2 lazy). Commit shared fixes HERE, then merge.
   |-- deploy/serverless serverless startup + config (this file, Dockerfile.v34*, BUILD_V34.md)
   `-- deploy/pod        pod startup, /workspace models, pod job API (DEPLOY_POD.md)
```

Updating a deploy branch after a common fix: `git checkout deploy/serverless && git merge perf/5090-speedup`.

## 1. Current serverless setup

- Worker: serverless handler (`/entrypoint.sh` -> `handler.py` -> `runpod.serverless.start`).
- Models: copied from the network volume (`/runpod-volume/models/<folder>/<file>`) to container disk at every
  cold start (`MODEL_SOURCE=volume`, the code default). ~29 GB, about 3 min of a cold start (R0 on 2026-11-03:
  177 s queue + cold start).
- Backend: parallel chunks on (`PARALLEL_CHUNKS_ENABLED=1`, `PARALLEL_MAX_CHUNKS=4`): up to 4 workers per video.

## 2. Image

| Image | Status |
|---|---|
| v32 `digamberjhazaran/infinitetalk-runpod@sha256:423a8e0c...0652` | base of v33/v34; the image the endpoint is believed to run. **Confirm the digest in the console.** |
| v33 (`Dockerfile.v33`) | not built |
| v34 (`Dockerfile.v34`, hf_cache default) | not built; steps in `BUILD_V34.md`, push to a PRIVATE repo only |
| `Dockerfile.v34.baked` | do not build |

v34 defaults to `MODEL_SOURCE=hf_cache` with volume fallback. To keep today's behaviour on a v34 image, set
`MODEL_SOURCE=volume` on the endpoint.

The public v32 repo contains an old layer with a RunPod key (dev scripts). The current key is a different one;
make sure the old key is revoked in the RunPod console.

## 3. Endpoint settings

| Setting | Value |
|---|---|
| GPU | RTX 5090 (32 GB). The worker rejects GPUs under 28 GB (`gpu_check`) |
| Active workers | 0 (each active worker is billed while idle) |
| Max workers | 4 or more (one per parallel chunk) |
| Idle timeout | short (5-60 s) unless warm hits matter more than idle billing |
| Execution timeout | at least 20 min (a 15 s chunk takes ~7-8 min warm) |
| FlashBoot | on |
| Network volume | attached, same datacenter; models under `models/<folder>/<file>` |
| Container disk | 60 GB+ (the volume copy lands on it) |
| Model (cached models) | empty for `MODEL_SOURCE=volume`; `<org>/infinitetalk-models` for hf_cache (see BUILD_V34.md) |

## 4. Endpoint env vars

| Var | Value | Note |
|---|---|---|
| `MODEL_SOURCE` | `volume` | v34 image default is `hf_cache` |
| `TORCH_COMPILE` | `1` | default |
| `COMPILE_CACHE` | `1` | compile cache on the volume (`/runpod-volume/cache/compile`) |
| `FP8_FAST` | `0` | only after the R2/R3 quality test |
| `UPSCALE_MODELS_AT_STARTUP` | unset (= 0) | SeedVR2 on the first upscale job only |
| `COMFYUI_MANAGER` | unset (= 0) | no startup network calls |
| `HF_CACHE_REPO` / `HF_CACHE_FALLBACK` | unset / `1` | hf_cache only |

Backend env (backend repo, `deploy/serverless`): `RUNPOD_API_KEY`, `ENDPOINT_ID`, `PARALLEL_CHUNKS_ENABLED=1`,
`PARALLEL_MAX_CHUNKS=4`, no `POD_BASE_URL`.

## 5. Back from pod to serverless

The pod keeps models at `/workspace/models/<folder>/<file>`. Serverless mounts the same network volume at
`/runpod-volume`, where `setup_models.py` looks for `models/<folder>/<file>`: **the same volume works for both**
when it is in the endpoint's datacenter.

1. Stop the pod (keep the network volume). A running pod and a serverless worker on the same volume is fine,
   but the pod bills while it runs.
2. Worker: `git checkout deploy/serverless`. Nothing to deploy if the endpoint image did not change.
3. Endpoint: attach the network volume, Max workers back to 4+, env as in section 4.
4. Backend: `git checkout deploy/serverless`, then in `backend/.env` remove `POD_BASE_URL` / `POD_API_TOKEN`,
   set `PARALLEL_CHUNKS_ENABLED=1`, keep `RUNPOD_API_KEY` + `ENDPOINT_ID`. Restart the backend
   (`docker compose up -d --build` on EC2).
5. Smoke test: `/api/health` shows the serverless workers block; one short single video; then one parallel video.
6. If a video fails: worker log in the RunPod console (`[MODELS]`, `[JOB] done`, `[TIMING] stages`).
