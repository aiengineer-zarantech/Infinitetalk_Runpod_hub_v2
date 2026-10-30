# Testing v20: torch.compile speed and quality (RTX 5090)

Goal: 1 minute of video in under 15 minutes on one RTX 5090 (32 GB), with no visible quality change.
Steps, cfg, shift, scheduler, seed, resolution, LoRA strength and all sampler settings are unchanged.
The only difference between the two benchmark runs is torch.compile.

## 1. Test endpoint

- Image: `digamberjhazaran/infinitetalk-runpod:v20` (pin the digest printed after the push)
- GPU: RTX 5090. Execution timeout: at least 3600 s.
- Env: leave `TORCH_COMPILE` and `FP8_FAST` unset (defaults: compile on, fp8_fast off).

## 2. Build the two payloads

From `D:\Serverless2\Infinitetalk_Runpod_hub_v2` (any Python 3, standard library only):

```powershell
python scripts\make_benchmark_payloads.py --image ..\face.jpeg --audio ..\en_temp.wav --width 720 --height 480
```

This writes `bench_compile_off.json` and `bench_compile_on.json`. Their exact content:

```json
{"input": {"benchmark": true, "torch_compile": false,
           "image_base64": "<base64 of face.jpeg>", "wav_base64": "<base64 of the audio>",
           "prompt": "A person talking naturally", "width": 720, "height": 480}}
```

```json
{"input": {"benchmark": true, "torch_compile": true,
           "image_base64": "<base64 of face.jpeg>", "wav_base64": "<base64 of the audio>",
           "prompt": "A person talking naturally", "width": 720, "height": 480}}
```

- `"benchmark": true` cuts the audio to the first 10 s (250 frames = 4 windows of 81 frames with 9 overlap) and allows the per-job `torch_compile` switch.
- Image, audio, prompt and size must be identical in both files. The workflow seed is fixed (2).

## 3. Run both jobs on the test endpoint

```powershell
$env:RUNPOD_API_KEY = "<your key>"   # in your own shell only, never in files
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload bench_compile_off.json -Out off.json
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload bench_compile_on.json  -Out on.json
```

Run them one after the other. The first compile-on job on a worker includes the compile time;
run `bench_compile_on.json` a second time on the same warm worker to see the steady-state speed.

## 4. Log lines to look for (RunPod worker logs; numbers below are examples)

| Line | What to check |
|---|---|
| `[GPU] OK: NVIDIA GeForce RTX 5090, 31.4 GB, CUDA 12.8` | Startup GPU check passed |
| `[OFFLOAD] auto: VRAM 31.4 GB, 720x480 -> blocks_to_swap 0, force_offload False` | 480p runs fully on the GPU |
| `[BENCHMARK] audio trimmed to the first 10s` | Benchmark mode active |
| `[COMPILE] torch.compile on; fp8_fast off` then `[COMPILE] using I2V_single_compiled.json` | Compile run (`off (job input)` for the off run) |
| `[SPEED] window 2/4: 41.3s, 6.10 s/it` | Per-window time and steady-state s/it |
| `[COMPILE] failed: ...` | Compile broke; the job re-ran once without it (`compile_fallback yes`) |
| `[JOB] done in ...` | Summary, see below |

The `[JOB] done` line contains: `gpu`, `vram`, `peak`, `blocks_to_swap`, `torch_compile on/off`,
`compile ~Xs` (first-step delay minus one normal step: compile plus moving the model to the GPU),
`fp8_fast on/off`, `avg ... s/it`, `sampling ...s` and `benchmark 10s audio`.

Compare `avg s/it` between the off and on runs. For the 15-minute target, a 1-minute video is about
21 windows of 6 steps, so you need roughly 7 s per step including each window's VAE decode
(`sampling` / number of steps).

## 5. Quality check (Windows)

Needs ffmpeg (`winget install Gyan.FFmpeg`). Standard library Python only.

```powershell
python scripts\compare_videos.py off.json on.json --out compare_out
```

- Inputs can be the saved RunPod responses (`off.json`, `on.json`) or `.mp4` files.
- PASS if the average SSIM is at least 0.97 and no single frame is below 0.90, with the same frame count and resolution.
- Exit code: 0 PASS, 1 FAIL, 2 could not compare.
- Outputs in `compare_out\`:
  - `side_by_side_off_vs_on.mp4`: left = compile off, right = compile on, with audio. Check the lip-sync by eye.
  - `ssim_per_frame.csv`: SSIM for every frame, to find where any difference is.

torch.compile is not bit-for-bit identical (fused kernels round slightly differently), so expect SSIM
very close to 1.0, not exactly 1.0.

## 6. Full-length run

Same as `bench_compile_on.json` but with `"benchmark"` and `"torch_compile"` removed and a 1-minute audio.
Check `[JOB] done in ...` against the 15-minute target.

## Switches (endpoint env)

| Env | Default | Effect |
|---|---|---|
| `TORCH_COMPILE` | `1` | `0` queues the original workflows exactly as before |
| `FP8_FAST` | `0` | `1` sets node 122 quantization to `fp8_e4m3fn_fast`. Changes numerics: test only |
| `OFFLOAD_TIERS` | built-in (offloading off) | JSON override of the VRAM tiers |
| `MIN_VRAM_GB` | `28` | GPUs below this are rejected at startup and per job (`refresh_worker`) |
| `OOM_EMERGENCY_BLOCKS` | `6` | Blocks swapped in the one emergency retry after an OOM (`force_offload False`) |
| `OFFLOAD_ALLOW_MANUAL` | `0` | `1` lets `"offload_mode": "manual"` jobs use their own `blocks_to_swap` / `force_offload` (tests only) |

From v21 offloading is off by default (`blocks_to_swap 0`, `force_offload False`), except 720p on GPUs below 44 GB
(32 GB cards): `blocks_to_swap 25`, `force_offload True`, because 720p without offloading ran out of memory on an RTX 5090.
If you ever see `[OFFLOAD] emergency swap after OOM` in the logs, that job ran out of VRAM and was retried once
(0 blocks -> 6, otherwise 10 more blocks, max 40).

From v23, 480p (width x height <= 900 x 500) on a GPU with 28 GB or more **always** queues `blocks_to_swap 0`,
`force_offload False`, `load_device main_device`. The client, `"offload_mode": "manual"` and `OFFLOAD_TIERS` cannot
change that; only the emergency OOM retry can. `MIN_VRAM_GB` does not move this 28 GB line.

## v23: final offload check before every /prompt

`handler.queue_prompt` checks the workflow right before it is sent to ComfyUI, also with `WORKER_GUARD_ENABLED=0`:

| Line | Meaning |
|---|---|
| `[OFFLOAD] final: blocks_to_swap 0, force_offload False, node 134 {'blocks_to_swap': 0, 'offload_img_emb': False, 'offload_txt_emb': False}, load_device main_device` | Logged for every video job: exactly what ComfyUI receives |
| `[OFFLOAD] final workflow has blocks_to_swap 38, force_offload True, ... Fixed before queueing` (ERROR) | The workflow did not match the job's offload decision (or broke the 480p rule) and was corrected. This should never appear; if it does, report it |

With the guard on, the workflow must match the job's decision (`[OFFLOAD] auto: ...` / `manual: ...`, or the
emergency retry). With the guard off, only the 480p rule is enforced, using the VRAM from `/system_stats`
(if that call fails, the GPU is assumed to have 28 GB or more).

Measured so far (for the 720p diagnosis below):
- 480p, 854x480: 20.1 to 22.1 GB on MIG 2g.48gb during sampling; out of memory on MIG 1g.24gb (23.6 GB) with ~22 GB
  allocated. So the 480p peak is about 24 to 26 GB.
- 720p, RTX 5090 (31.4 GB), blocks_to_swap 0, compile on: 21.6 GB during steps 1-4 of window 1 (2026-10-27 ~12:35 UTC),
  then an OOM later in the job (owner report, no log). Suggests the transformer forward fits and the OOM is in the
  per-window VAE decode/encode, which runs **inside** node 128 (WanVideoSampler) in InfiniteTalk mode. Only the
  traceback can confirm it.

## Step 2: where does 720p run out of memory on a 32 GB RTX 5090?

Until this is measured, the 720p row on 28-44 GB stays at `blocks_to_swap 25`, `force_offload True`.

### Endpoint

- Image: `digamberjhazaran/infinitetalk-runpod:v23` (contains the v22 profiler, off unless `PROFILE=1`)
- GPU: RTX 5090. Use a test endpoint, not production.
- Env for 2b, 2c, 2d: `OFFLOAD_ALLOW_MANUAL=1`, `PROFILE=1`. Optional: `VRAM_SAMPLE_SEC=1` for a finer peak (default 5 s).
- Env for 2a: none needed.

### Payloads

All four are in `payloads\step2\`. To rebuild them (same image and audio as the A/B benchmark):

```powershell
python scripts\make_step2_payloads.py --from ..\.baselines\bench\bench_compile_off.json
```

Run each one with:

```powershell
$env:RUNPOD_API_KEY = "<your key>"   # in your own shell only, never in files
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload payloads\step2\step2b_repro_720p.json -Out step2b.json
```

Order: **2a**, then **2b**, then **either 2c or 2d** depending on the 2b traceback.

### 2a. `step2a_object_info.json`: installed wrapper inputs (cmd job, no GPU work)

```json
{
 "input": {
  "cmd": "cd /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && echo '== wrapper commit' && git log -1 --format='%h %ci %s' 2>&1; python3 -c \"import json, urllib.request\nfor n in ('WanVideoImageToVideoMultiTalk', 'WanVideoDecode', 'WanVideoEncode', 'WanVideoVAELoader', 'WanVideoSampler'):\n    d = json.load(urllib.request.urlopen('http://127.0.0.1:8188/object_info/' + n, timeout=30)).get(n, {})\n    print('== ' + n + (' (NOT INSTALLED)' if not d else ''))\n    for kind in ('required', 'optional'):\n        for k, v in d.get('input', {}).get(kind, {}).items():\n            print('   ', kind, k, json.dumps(v)[:200])\n\" 2>&1; echo '== tiled_vae in wrapper source'; grep -rn --include=*.py 'tiled_vae' . | head -60; echo '== vae.decode / vae.encode calls'; grep -rn --include=*.py -E 'vae\\.(decode|encode)\\(' . | head -60"
 }
}
```

The result is in `output` of the saved response. Look for:

| Output | What it confirms |
|---|---|
| `== WanVideoImageToVideoMultiTalk` then a line `required tiled_vae ["BOOLEAN", ...]` (or `optional`) | Node 192's tiling input is called `tiled_vae` |
| `== WanVideoDecode` then `enable_vae_tiling`, `tile_x`, `tile_y`, `tile_stride_x`, `tile_stride_y` | Node 130's tiling input names |
| `== WanVideoEncode` then `enable_vae_tiling` | V2V node 229's tiling input name |
| `== tiled_vae in wrapper source`: a line in the sampler / multitalk loop that passes `tiled_vae` to `vae.decode(...)` or `vae.encode(...)` | The per-window decode/encode inside node 128 uses node 192's `tiled_vae` |
| `(NOT INSTALLED)` after a node name, or other input names | Stop: v23 sets `tiled_vae` / `enable_vae_tiling`, so 2c needs a fix first |

### 2b. `step2b_repro_720p.json`: reproduce the OOM with no offloading

```json
{"input": {"benchmark": true, "torch_compile": false, "offload_mode": "manual",
           "blocks_to_swap": 0, "force_offload": false, "width": 1280, "height": 720,
           "prompt": "A person talking naturally",
           "image_base64": "<base64 of the benchmark image>", "wav_base64": "<base64 of the benchmark audio>"}}
```

Manual jobs get no emergency retry, so the OOM is reported, not hidden. Log lines (RunPod worker log; ComfyUI's own
output is prefixed `[comfy]`):

| Line | What to read |
|---|---|
| `[OFFLOAD] manual: blocks_to_swap 0, force_offload False (as sent)` | Manual mode accepted (needs `OFFLOAD_ALLOW_MANUAL=1`) |
| `[OFFLOAD] final: blocks_to_swap 0, force_offload False, node 134 {...}, load_device main_device` | ComfyUI really got no offloading |
| `[BENCHMARK] audio trimmed to the first 10s`, `[COMPILE] torch.compile off (job input)` | 10 s audio (4 windows), no compile |
| `[SPEED] window k/4: ...` | Window k finished all 6 steps |
| `[WATCHDOG] VRAM used X / 31.4 GB` | VRAM every 30 s |
| `[PROFILE] window k/n: steps ..., vae_decode ..., vae_encode ...` and `[PROFILE] loop: n windows in ...s (loop raised an exception; windows finished so far)` | n windows finished; the OOM happened in window **n + 1** |
| `❌ ComfyUI execution_error at node 128 (WanVideoSampler): torch.OutOfMemoryError: ...` followed by the traceback | **Node**, and the **allocation size** if torch put it in the message (`Tried to allocate X GiB` / `Requested ...`). If the message is only `Allocation on device`, use the `[comfy]` lines printed just before it |
| `[JOB] failed in ... peak X GB ...` | **Peak VRAM** before the OOM (sampled every `VRAM_SAMPLE_SEC`) |

Where it failed, from the traceback's last wrapper frames:
- `vae.decode(` / `vae.encode(` / `wan_video_vae.py`: the per-window **VAE** step. Run **2c**.
  Extra hint: a `[SPEED] window k/4` line for the failing window means all its steps had finished.
- The transformer forward (WanModel blocks, attention): the **sampler**. Run **2d**.

Report: node, window, allocation size, peak VRAM, and the traceback's last 10 lines.

### 2c. `step2c_vae_tiling_720p.json`: only if 2b failed in the VAE

```json
{"input": {"benchmark": true, "torch_compile": false, "offload_mode": "manual",
           "blocks_to_swap": 0, "force_offload": false, "width": 1280, "height": 720,
           "prompt": "A person talking naturally",
           "image_base64": "<base64 of the benchmark image>", "wav_base64": "<base64 of the benchmark audio>",
           "vae_tiling": true}}
```

`"vae_tiling": true` (v23 test input) sets node 192 `tiled_vae` and node 130 `enable_vae_tiling` (and node 229 in V2V)
to `true`; nothing else changes. Only run it if 2a confirmed those names.

| Line | What to check |
|---|---|
| `✅ 노드 192 (WanVideoImageToVideoMultiTalk) VAE tiling on` and `✅ 노드 130 (WanVideoDecode) VAE tiling on` | Tiling was applied ("노드" = node) |
| `[OFFLOAD] final: blocks_to_swap 0, force_offload False, ...` | Still no offloading |
| `[PROFILE] window k/4: ... vae_decode ...` | VAE time per window with tiling (compare with 2b's finished windows) |
| `[JOB] done in ... peak X GB ...` | Success, and the peak. Fix is good if peak + 2 GB <= 31.4 GB |

Tiled decoding can leave faint seams between tiles. Check the video by eye before making it the default.

### 2d. `step2d_measure_25_blocks.json`: only if 2b failed in the transformer forward

```json
{"input": {"benchmark": true, "torch_compile": false, "offload_mode": "manual",
           "blocks_to_swap": 25, "force_offload": false, "width": 1280, "height": 720,
           "prompt": "A person talking naturally",
           "image_base64": "<base64 of the benchmark image>", "wav_base64": "<base64 of the benchmark audio>"}}
```

| Line | What to read |
|---|---|
| `[OFFLOAD] final: blocks_to_swap 25, force_offload False, node 134 {'blocks_to_swap': 25, 'offload_img_emb': True, 'offload_txt_emb': True}, load_device offload_device` | 25 blocks swapped |
| `[comfy] ... [Sampling] Max allocated memory: max_memory=X GB` | **X**, the sampling peak with 25 blocks off the GPU |
| `[JOB] done in ... peak Y GB ...` | Device-level peak, for comparison |

Minimum blocks with ~2 GB headroom (one block ~0.38 GB):

```
peak_at_0 = X + 25 * 0.38
blocks    = ceil((peak_at_0 + 2 - 31.4) / 0.38), at least 0
```

Example: X = 22.0 GB gives peak_at_0 = 31.5 GB and blocks = ceil(2.1 / 0.38) = 6.

## Phase A (v24): full GPU for 480p and fast480, resident model, 720p calibration, OOM ladder

Branch `feature/fast480-upscale`. Old jobs are unchanged: `tests/test_v23_prompt_identity.py` runs the same jobs
(what the studio sends today, plus 720p, V2V, multi, compile off, FP8_FAST, benchmark, manual, VAE tiling) through
the handler and worker_guard of `v1.0-stable` and of v24 and checks that every workflow sent to ComfyUI and every
result is identical. New behaviour only starts with the new `"mode": "fast480"` input, an OOM, or the env switches below.

### Endpoint

- Image: `digamberjhazaran/infinitetalk-runpod:v24`, digest `sha256:2ca2170389752fbae558a9e54997fe04f2d62f9d654e60b37e08abde06af3870`
- GPU: RTX 5090 (31.4 GB). Use a test endpoint, not production. Max workers 1 for A.3 (both jobs must reach the same warm worker).
- **Execution timeout: 3900 s** (endpoint settings, Execution Timeout), and env **`RUNPOD_EXECUTION_TIMEOUT_SEC=3900`**
  with the same number. RunPod does not tell the worker its timeout, so the worker needs the env to know it.
  Check the endpoint's current value: I believe the RunPod default is 600 s, far too short for long 720p jobs.
  With `JOB_TIMEOUT_SEC` at its default 3600, the worker stops a job itself at 3600 s with a clear error, 300 s
  before RunPod would kill it. The worker uses the smaller of `JOB_TIMEOUT_SEC` and
  `RUNPOD_EXECUTION_TIMEOUT_SEC - 120` (`RUNPOD_RETURN_MARGIN_SEC`: time to encode and return the video).
  Without the env the worker logs `[GUARD] RUNPOD_EXECUTION_TIMEOUT_SEC not set: ...` at startup.
- Env for A.4: `OFFLOAD_ALLOW_MANUAL=1`. Optional for A.4: `PROFILE=1` (per-window times; adds GPU syncs).
- Env for A.5 only: see A.5. Remove it afterwards.

### Payloads

All in `payloads\phaseA\` (git-ignored: they contain the image and audio as base64). Same image, audio and prompt
as the A/B benchmark. Every job is a benchmark job, so the worker cuts the audio to 10 s (4 windows). The seed is the
workflow's fixed 2. Rebuild:

```powershell
python scripts\make_fast480_payloads.py --from ..\.baselines\bench\bench_compile_off.json
```

Run each one with:

```powershell
$env:RUNPOD_API_KEY = "<your key>"   # in your own shell only, never in files
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload payloads\phaseA\a2_fast480_16x9.json -Out a2_fast480_16x9.json
```

| File | Input (besides image, audio, prompt, `"benchmark": true`) |
|---|---|
| `a2_normal_480p.json` | `"width": 854, "height": 480` |
| `a2_fast480_16x9.json` | `"mode": "fast480"` (848x480) |
| `a2_fast480_9x16.json` | `"mode": "fast480", "aspect": "9:16"` (480x848) |
| `a4_720p_b0_tiling_off.json`, `a4_720p_b5_tiling_off.json`, `a4_720p_b10_tiling_off.json` | `"width": 1280, "height": 720, "offload_mode": "manual", "blocks_to_swap": 0 / 5 / 10, "force_offload": false, "vae_tiling": false` |
| `a4_720p_b0_tiling_on.json`, `a4_720p_b5_tiling_on.json`, `a4_720p_b10_tiling_on.json` | same with `"vae_tiling": true` |
| `a5_ladder_720p.json` | `"width": 1280, "height": 720` (auto mode) |

### A.2: 480p and fast480 run fully on the GPU

Run `a2_normal_480p.json`, `a2_fast480_16x9.json`, `a2_fast480_9x16.json`.

| Line | What to check |
|---|---|
| `[OFFLOAD] auto: VRAM 31.4 GB, 848x480 -> blocks_to_swap 0, force_offload False` | Decision (854x480 / 480x848 for the others) |
| `[OFFLOAD] final: blocks_to_swap 0, force_offload False, node 134 {'blocks_to_swap': 0, 'offload_img_emb': False, 'offload_txt_emb': False}, load_device main_device` | **The proof**: exactly what ComfyUI received |
| `[MODE] fast480: width/height ... replaced by 848x480` | Only if the client sent another size |
| `[VRAM] vram peak sampler X GB, vae Y GB, headroom Z GB (...)` | Per-stage peaks |
| `[JOB] fast480 done: res 848x480, gen Xs, upscale n/a, total Zs, attention sageattn, blocks_to_swap 0, vram peak sampler A GB / vae B GB / upscale n/a` | Final line. `upscale` stays `n/a` until Phase D |

The 9:16 output must be 480x848 with the face center-cropped, not stretched (the workflow's ImageResizeKJv2 node
crops to the target size from the center).

### A.3: the model stays in VRAM between jobs

Right after A.2, on the same warm worker, run `a2_fast480_16x9.json` once more.

| Line | What to check |
|---|---|
| `[RESIDENT] video job #N on this worker: model loader 122 reused from VRAM (cached); VRAM used at job start X GB` | **The proof**: ComfyUI reused the loaded model. X should be well above 1 GB (the model is already on the GPU) |
| `[RESIDENT] ... model loader 122 loaded` | First job on a worker, or the model settings changed |
| `[RESIDENT] model settings changed (nodes ...); unloading the previous model before queueing` | A job needs another model (blocks_to_swap, compile setting, later the attention mode). The old one is unloaded first so two copies never share the VRAM |

The time before the first `[SPEED] window 1/4` line should also be much shorter on the second run.

### A.4: 720p at blocks_to_swap 0, 5 and 10 (calibration)

Env `OFFLOAD_ALLOW_MANUAL=1`. Order (smallest risk first): `b10_tiling_off`, `b5_tiling_off`, `b0_tiling_off`,
`b10_tiling_on`, `b5_tiling_on`, `b0_tiling_on`. Manual jobs get no OOM ladder, so an OOM is reported, not hidden.
Production 720p runs without VAE tiling, so the tiling-off numbers are the ones the default is based on.

Send me, for each job:

| Line | Why |
|---|---|
| `[OFFLOAD] manual: blocks_to_swap N, force_offload False (as sent)` and `[OFFLOAD] final: ...` | The job really ran with N |
| `[comfy] Transformer blocks on cpu: X MB` (in the "Block swap memory summary"; b5 and b10 only) | **Exact size of one block** = X / N. Replaces the 0.44 GB estimate |
| `[VRAM] vram peak sampler X GB, vae Y GB, headroom Z GB` | Peaks per stage at N |
| `[SPEED] window k/4: ...` and `[JOB] standard done: ...` | Speed cost of swapping |
| On failure: `ComfyUI execution_error at node 128 ...` and the traceback's last 10 lines | Where 0 blocks runs out of memory |

From these I fit the adaptive calibration (block size, sampler and VAE peak per resolution) and set the 720p
default to the lowest blocks_to_swap with at least 2 GB headroom (never 25 unless measured). That goes into the
next image; until then 720p on 28-44 GB stays at 25 / True. A calibration can also be tried without a rebuild:
`OFFLOAD_CALIBRATION={"block_gb": B, "sampler": {"base_gb": a, "gb_per_ktoken": b}, "vae": {"base_gb": c, "gb_per_ktoken": d}}`.

### A.5: OOM ladder, failing stage, RAM

Only meaningful if `b0_tiling_off` ran out of memory in A.4. Set this env on the test endpoint (720p then starts at
0 blocks in auto mode; the 480p rule is not affected), run `a5_ladder_720p.json`, then **remove the env**:

```
OFFLOAD_TIERS=[{"min_vram_gb": 28, "480p": {"blocks_to_swap": 0, "force_offload": false}, "720p": {"blocks_to_swap": 0, "force_offload": false}}]
```

| Line | What to check |
|---|---|
| `[OFFLOAD] RAM (job): limit X GB, used Y GB, available Z GB (cgroup v2); blocks_to_swap N needs ~A GB, 40 needs ~B GB (block 0.44 GB, estimate; unpinned RAM)` | Container RAM, and whether the last ladder steps (40 blocks) fit on this worker type. Logged for every video job |
| `[OFFLOAD] RAM: OOM ladder steps above K blocks would not fit ...` (WARNING) or `[OFFLOAD] RAM too low for blocks_to_swap N ...` (ERROR) | Only when RAM is short |
| `[OFFLOAD] OOM retry 1/4: out of memory at vae_decode (node 128 WanVideoSampler, window 3); blocks_to_swap 0 -> 10, force_offload False (~Ns needed, Ms left)` | **Failing stage** of the attempt that ran out of memory (`sampler (transformer steps, ...)` when it was the transformer) |
| `[OFFLOAD] RAM (OOM retry k): ...` | RAM for that step |
| `[OFFLOAD] OOM retry final: VAE tiling ON (output may differ)` | Only after the 40-block attempt also ran out of memory. The output then has `"emergency_vae_tiling": true` |
| `[OFFLOAD] OOM retry k/4 not started: needs ~Ns (...), only Ms left before ...` | Not enough time for another attempt (estimate: windows done so far x 1.5, or 2x a failed attempt, at least `OOM_RETRY_MIN_SEC` = 120 s) |
| `[OFFLOAD] adaptive: predicted peak n/a (not calibrated, tier value used), chosen blocks_to_swap N, actual peak Z GB (W%), budget ...` | Every job. The prediction appears once calibrated |

Ladder: +10, then 25, then 40 (only values above the step before), then 40 with VAE tiling. Normal jobs never get
VAE tiling from the worker; the old opt-in `"vae_tiling": true` job input (test input since v23) is unchanged.

### A.6 (optional): allocator with expandable segments

Env `ALLOC_MODE=native_expandable`: ComfyUI starts with `--disable-cuda-malloc` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (with ComfyUI's default cudaMallocAsync backend, torch ignores this
option). Run `a2_fast480_16x9.json` and `a4_720p_b10_tiling_off.json` and compare with the runs above:

| Line | What to check |
|---|---|
| `ALLOC_MODE=native_expandable: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, ComfyUI --disable-cuda-malloc (test only)` | Switch active |
| `[comfy] Device: cuda:0 NVIDIA GeForce RTX 5090 : native` | ComfyUI uses torch's native allocator (default runs say `cudaMallocAsync`) |
| `[VRAM] ...` and `[SPEED] ...` | Peaks and speed against the default allocator |
| torch warning `PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead` | Expected with torch 2.9.1, harmless |

Frames should be identical (the allocator does not change numerics). Check with the framemd5 commands in
PROFILING.md section 5. If they differ, first check that two runs with the default allocator are identical.

### New and changed env (all optional)

| Env | Default | Meaning |
|---|---|---|
| `RUNPOD_EXECUTION_TIMEOUT_SEC` | unset | The endpoint's execution timeout, for the OOM ladder time check and the watchdog |
| `RUNPOD_RETURN_MARGIN_SEC` | 120 | Kept free at the end of the RunPod limit |
| `OOM_RETRY_MIN_SEC` | 120 | Minimum time a ladder step needs |
| `OFFLOAD_CALIBRATION` | unset | Adaptive calibration JSON (see A.4) |
| `OFFLOAD_TARGET_FRACTION` | 0.85 | Share of the VRAM the predicted peak may use (0.5 to 0.95) |
| `VRAM_STAGES` / `VRAM_STAGE_INTERVAL_MS` | 1 / 50 | Per-stage VRAM sampler inside ComfyUI; 0 turns it off |
| `ALLOC_MODE` | unset | `native_expandable`, test only (A.6) |
| `OOM_EMERGENCY_BLOCKS` | removed | Replaced by the ladder |

## v25: negative prompt with NAG (head motion)

New optional input `"negative_prompt"`. The sampler runs at cfg 1.0 (lightx2v distill LoRA), which skips the
negative pass, so the worker injects `WanVideoApplyNAG` (node 900, fed by node 241's negative-only output) and
points the sampler's `text_embeds` at it. Jobs without `negative_prompt` queue exactly the same workflow as
v1.0-stable (`tests/test_v23_prompt_identity.py`); `tests/test_negative_prompt_nag.py` covers the new path.
Optional: `"nag_scale"` (11.0), `"nag_tau"` (2.5), `"nag_alpha"` (0.25). A `|` multi-prompt keeps NAG off.

### Endpoint

- Image: `digamberjhazaran/infinitetalk-runpod:v25`, digest `sha256:cbb4e2e209562175033d972dd9a85b8e116caa4e1e3cc7e1ee6e3d4e70c850af`
- Same test endpoint settings as Phase A.

### A/B payloads

Same image, audio, prompt and seed (workflow seed 2). B is A plus the Studio default negative prompt:

```powershell
$a = Get-Content payloads\phaseA\a2_normal_480p.json -Raw | ConvertFrom-Json
$a | ConvertTo-Json -Depth 10 | Set-Content payloads\nag_a_off.json -Encoding utf8
$a.input | Add-Member negative_prompt "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, excessive head movement, head shaking, nodding, swaying, turning head, tilting head, camera shake"
$a | ConvertTo-Json -Depth 10 | Set-Content payloads\nag_b_on.json -Encoding utf8
```

### What to check

- Logs of B: `[NAG] on: scale=11.0, tau=2.5, alpha=0.25` and `네거티브 프롬프트(241) 설정: ...`. A has no `[NAG]` line.
- A runs as before (same frames as v24 for the same payload).
- B vs A: less head shaking, nodding and swaying; lip sync and face identity unchanged; note the generation time
  difference (NAG adds one extra cross-attention per block, no extra sampler pass).
- If B looks stiff or over-corrected, try `"nag_scale": 7`; if motion is still too strong, try 15.

## v26: fast480 + SeedVR2 720p upscale, upscale_only

`"mode": "fast480"` now generates at 848x480 / 480x848 and then upscales to exactly 1280x720 / 720x1280 with
SeedVR2 (the native nodes of the installed ComfyUI core, no new package). `"mode": "upscale_only"` runs only the
upscale on an existing video. Jobs without `mode` / `upscale` are unchanged (`tests/test_v23_prompt_identity.py`).

How the stage works (worker_guard/upscale.py):
1. InfiniteTalk is unloaded (`/free`, wait until VRAM is near zero).
2. The video is upscaled in time segments sized from the container RAM. Each segment loads 8 extra frames on each
   side for context, only its core frames are kept. One ComfyUI prompt per segment (`Upscale_SeedVR2.json`).
3. ffmpeg joins the core frames (H.264 crf 19, yuv420p) and copies the source audio unchanged (`-c:a copy`).
4. Checks: every segment returns exactly the frames it was given; the result has the source's frame count, frame
   rate and audio packets and exactly the target size. Any difference fails the upscale.
5. Frozen-frame check (log only), then SeedVR2 is unloaded too.

If the upscale fails (out of memory after 3 smaller retries of that segment, time limit, failed check), a fast480 job
still returns its 480p video with `"upscale": "failed"` and `"upscale_error"`. InfiniteTalk is never re-run for it.
A successful fast480 job returns `"upscale": "720p"` next to `"video"`.

### Endpoint

- Image: `digamberjhazaran/infinitetalk-runpod:v26`, digest `<filled in after the push>`
- Same test endpoint settings as Phase A (RTX 5090, execution timeout 3900 s, `RUNPOD_EXECUTION_TIMEOUT_SEC=3900`).
- Volume: SeedVR2 files go to `/runpod-volume/models/upscale/`. If they are not there, the worker downloads them once
  from `huggingface.co/Comfy-Org/SeedVR2` (revision `df48879`, sha256 checked):
  - 3B fp16 (6.8 GB) + VAE (0.5 GB) during the first startup (that cold start takes longer once),
  - 7B fp16 (16.5 GB) during the first job that asks for it (that job takes longer once).
  Local disk needs about 7.3 GB more, plus 16.5 GB once 7B is used.

### Payloads

Pick an existing 480p InfiniteTalk clip (848x480, with audio) and a 15-20 s voice:

```powershell
python scripts\make_upscale_payloads.py --video clip480.mp4 --seconds 10 --image face.jpeg --audio voice_18s.wav
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload payloads\v26\u1_upscale_only_3b.json -Out u1_upscale_only_3b.json
```

(`--seconds` needs ffmpeg on PATH; otherwise cut the clip yourself. `--from <payload>` instead of `--image` +
`--audio` reuses that payload's image, audio and prompt.)

### Test order

1. **u1_upscale_only_3b.json** (10 s existing clip, 3B). This decides whether SeedVR2 is usable for talking heads.
2. **u2_upscale_only_7b.json** (same clip, 7B). First 7B job downloads/copies the model.
3. **u3_fast480_16x9.json**, then **u4_fast480_9x16.json** (full jobs: generation + upscale).
4. Step 3 comparison (direct 720p vs 480p + upscale vs 480p): payloads and compare script come with Step 3.

### Log lines to look for

```
[UPSCALE] source 848x480, 250 frames at 25/1 fps, audio yes -> 1280x720, model seedvr2_3b_fp16.safetensors
[UPSCALE] models ready in 2.1s: seedvr2_3b_fp16.safetensors, seedvr2_ema_vae_fp16.safetensors
[UPSCALE] segments: 250 frames -> 1 segment(s) of 337 frames (+8 context each side), RAM available 56.0 GB, per frame 150 MB
[UPSCALE] segment 1/1: frames 0-249 (loaded 0-249) in 95.2s
[UPSCALE] frozen-frame check: 0 suspicious frames (of 250; by frame%4: 0/0/0/0; first: -)
[UPSCALE] model seedvr2_3b_fp16, input 848x480 -> output 1280x720, frames 250, time 110.4s, vram peak 18.2 GB
[UPSCALE] RAM peak 31.5 GB (limit 62.0 GB), segment 337 frames, 1 segment(s)
[JOB] upscale_only done: res 848x480 -> 1280x720, gen n/a, upscale 112.0s, total 113.5s, ...
[JOB] fast480 done: res 848x480 -> 1280x720, gen 140.2s, upscale 180.3s, total 330.1s, ..., vram peak sampler ... / upscale 18.2 GB
```
(Numbers are examples.)

What to check and send me:
- **Frozen-frame line** (go/no-go for ComfyUI issue #16591). Many suspicious frames concentrated in `frame%4` bucket 0
  means the issue reproduces: send me the line. Then watch those frames (the line lists the first ten indices).
- **Mouth, teeth, eyes** in the 720p result vs the 480p input; any flicker; colour vs the input.
- **Segment seams**: a 10 s clip is usually one segment. To see a seam, add `"upscale_segment_frames": 97` to u1 and look
  at frames around 97 and 194.
- **Times**: upscale time vs generation time; total vs your direct 720p jobs.
- **RAM peak** and **vram peak** lines: they calibrate `UPSCALE_RAM_PER_FRAME_MB` (segment size) and show the headroom.

### Optional inputs (tests only)

| Input | Default | Meaning |
|---|---|---|
| `"upscale": "none"` | `"720p"` | fast480 without upscale (the v24/v25 output) |
| `"upscale_model"` | `"3b"` | `"7b"` |
| `"upscale_segment_frames"` | from RAM | force the segment size (rounded to 4n+1) |
| `"upscale_chunk_frames"` | auto | force SeedVR2's temporal chunk (4n+1) for the first try |
| `"upscale_color_correction"` | `"lab"` | `"wavelet"`, `"adain"`, `"none"` |

### New env (optional)

| Env | Default | Meaning |
|---|---|---|
| `UPSCALE_RAM_PER_FRAME_MB` | 150 | RAM per 1280x720 frame used to size the segments (calibrated from the RAM peak line) |
| `UPSCALE_MODELS_AT_STARTUP` | 1 | 0 = prepare the upscale models on the first upscale job instead of at startup |
