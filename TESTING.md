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
