# MODEL_LOCK: load the model once per worker, never reload

Default on (`MODEL_LOCK=1`). `MODEL_LOCK=0` = the previous behaviour, for a separate diagnostics endpoint/pod only.

## Why

ComfyUI keeps a loaded model only while every input of the loader chain stays identical, and after any OOM it
unloads all models ("Got an OOM, unloading all loaded models"). Reloads seen in the logs came from:
blocks_to_swap changing on each OOM retry (0 -> 10 -> 25), switching between `I2V_single.json` and
`I2V_single_compiled.json` (compile fallback), and A/B jobs with other settings on production workers.

## What the lock does

| Rule | Where |
|---|---|
| One config: `locked_model_config.json` (workflow, compile, precision, quantization, LoRA + strength, attention, blocks_to_swap, load_device, offload flags, max size, VRAM model) | repo root, copied to `/` |
| Job fields that change a model setting are ignored and logged (`[MODEL_LOCK] ignored job fields ...`): blocks_to_swap, force_offload, load_device, base_precision, quantization, attention_mode, torch_compile, fp8_fast, merge_loras, lora*, vae_tiling, offload_*, benchmark, mode, upscale* | `model_lock.sanitize` |
| Jobs the lock cannot run are refused with a clear error: multi-person, video input, larger than 854x480 | `model_lock.sanitize` |
| Locked values written into nodes 122 / 138 / 134 / 128 before queueing (normally already equal) | `model_lock.enforce` |
| Resident signature = loader-chain signature of the last good job. `MODEL REUSED` / `MODEL LOADED (cold start)`; a different signature is refused (`refusing to load a second copy`), never a silent reload | `ResidentState`, `_locked_before_queue` |
| After the job: verified against ComfyUI's `execution_cached` (loader node 122 cached = really reused); an unexpected reload is logged as an error | `_locked_after` |
| VRAM pre-flight: projected peak = sampler peak + final-decode cost per frame + non-model memory now; needs `headroom_gb` free. Otherwise: clear non-model memory (`POST /worker/clear_cache`: gc + empty_cache, models stay loaded), then tiled final VAE decode (node 130), then fail fast with `refresh_worker` | `model_lock.preflight`, `comfy_mem/` |
| No OOM ladder, no compile fallback, no retry with other settings on the same worker. An OOM refreshes the worker and clears the resident state | `_locked_failure` |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`ALLOC_MODE=native_expandable`) unless `ALLOC_MODE` is set (`cuda_malloc` = ComfyUI default) | `entrypoint.sh` |
| Per job: `[MODEL] job N: MODEL REUSED|LOADED, VRAM start, peak, headroom, blocks_to_swap, tiled_vae, frames, time`; the same in `output.model` for the backend/UI | `_locked_after` |

`/free` is never used under the lock: its `free_memory` resets ComfyUI's execution cache, which drops the cached
model too.

## The locked blocks_to_swap

Measured 2026-11-03 on pod b98p0yq88j1lel (RTX 5090, 31.4 GB, v32 code, 854x480, blocks_to_swap 0, compile on,
1615 frames / 22 windows): watchdog peak 22.4 GB while sampling, stage peak 25.9 GB = **5.5 GB headroom**.
0 is the lowest possible value and already leaves more than 2.5 GB, so `blocks_to_swap = 0` is locked.

`decode_gb_per_frame` (0.00217) is a one-point estimate from that run: (25.9 - 22.4) / 1615. With it a 2 min job
(3081 frames) projects 29.1 GB, below the 2.5 GB headroom, so the pre-flight turns on the tiled final decode.
**Re-measure with the 2 min job of the test below** and update `vram` in the config.

## Max frames (2026-11-03 pod test)

Job 1 (30 s, 831 frames, cold): COMPLETED in 994 s, peak 25.0 GB, headroom 6.3 GB, blocks_to_swap 0.
Job 2 (2 min, 3081 frames): `MODEL REUSED` (no reload), then OOM in the sampler before its first step (31.3 GB).
The sampler cost grows with the frame count, so the tiled-decode estimate above was wrong for long jobs.
`max_frames` = 1615 (61 s of audio, ran without an OOM) is now locked: longer jobs are refused before queueing
(no OOM, no worker refresh, the model stays loaded). The backend (deploy/pod) splits longer audio into <= 60 s
chunks that run one after another on the same pod.

## Separate diagnostics

Benchmark / A-B fields are ignored on a locked worker, so diagnostics cannot change a production model. Run them
on a separate endpoint or pod with `MODEL_LOCK=0`. The backend test scripts refuse the production endpoint
(`TEST_ENDPOINT_ID` must be set and differ from `ENDPOINT_ID`).

## GPU test (not run yet: needs this code on a worker)

5 normal jobs back to back on ONE worker, 848x480: 30 s, 2 min, 30 s, 2 min, 30 s.
Expected: job 1 `MODEL LOADED (cold start)`, jobs 2-5 `MODEL REUSED`, zero OOM, blocks_to_swap 0 every time,
headroom >= 2.5 GB in every `[MODEL] job N:` line. Report the `[MODEL] job N:` lines and `[JOB] done in` times.
