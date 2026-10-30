# Profiling v22: where the time between sampler steps goes

`PROFILE=1` logs, for every InfiniteTalk window, how long the steps, VAE decode, VAE encode and
everything else took. Off by default. No quality setting, package or workflow file changes.

## 1. Test endpoint (not production)

- Image: `digamberjhazaran/infinitetalk-runpod:v22`
- Env: `PROFILE=1` (keep all other env the same as the endpoint that showed ~67 s per window)
- GPU: the same type as that measurement (RTX 5090, 32 GB)

## 2. Payload

Use the same image, audio and size as the 67 s measurement. With the v20 tool:

```powershell
python scripts\make_benchmark_payloads.py --image face.jpeg --audio voice.wav --width 720 --height 480
```

Run `bench_compile_on.json` (the production default, torch.compile on). The audio is cut to 10 s, which gives 4 windows.

## 3. Run (live, billable GPU call)

```powershell
$env:RUNPOD_API_KEY = "<your key>"   # in your own shell only
.\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload bench_compile_on.json -Out profile_on.json
```

## 4. Log lines (RunPod worker logs)

From ComfyUI (prefixed `[comfy]`), one pair per window, then a summary:

```
[PROFILE] window 2/4: steps X.Xs, vae_decode Y.Ys, vae_encode Z.Zs, other W.Ws
[PROFILE] window 2/4 detail: total T.Ts; encode 81f a.aas + 9f b.bbs; decode 21 latent f c.ccs; other = vae_move d.dds (4x), empty_cache+gc e.ees, prep f.ffs, untracked g.ggs; first_step h.hhs, avg i.ii s/it (6 steps)
[PROFILE] loop: 4 windows in T.Ts; steps X.Xs (P%), vae_decode Y.Ys (P%), vae_encode Z.Zs (P%), other W.Ws (P%); after last window A.As
[PROFILE] windows 2..4 average: total T.Ts, steps X.Xs, vae_decode Y.Ys, vae_encode Z.Zs, other W.Ws (vae_move d.dds, empty_cache+gc e.ees)
```

From the handler, one line per node (audio embedding, CLIP vision, text encoding, loaders, video combine):

```
[PROFILE] node 194 (MultiTalkWav2VecEmbeds): ...s
[PROFILE] nodes: N ran in ...s; cached: ...
```

Meaning of the window fields:

| Field | What it covers |
|---|---|
| steps | from the step loop start to the VAE call that starts the decode: 6 transformer passes plus scheduler updates |
| vae_decode | `vae.decode` of the window (21 latent frames to 81 frames) |
| vae_encode | `vae.encode` of the 81-frame conditioning clip plus the motion frames (1 in window 1, 9 after) |
| other | the rest: VAE moves between CPU and GPU, `soft_empty_cache` + `gc.collect`, prep (previous window's frame slicing and audio padding, this window's audio slice and noise; loop setup in window 1), untracked |

Window 1 also includes torch.compile and first-use costs, so use the "windows 2..n average" line for the steady state.
The full data is also written to `/tmp/worker_profile/loop_*.json` inside the worker.

## 5. Optional: prove the profiler does not change the output

Set `PROFILE=0` on the test endpoint, let the workers restart, run the same payload to `profile_off.json`, then:

```powershell
python -c "import json,base64,sys; [open(o,'wb').write(base64.b64decode(json.load(open(i,encoding='utf-8-sig'))['output']['video'])) for i,o in (('profile_on.json','on.mp4'),('profile_off.json','off.mp4'))]"
ffmpeg -v error -i on.mp4 -map 0:v -f framemd5 on.md5
ffmpeg -v error -i off.mp4 -map 0:v -f framemd5 off.md5
fc.exe on.md5 off.md5
```

Identical files mean the frames are bit-identical. If they differ, first run the same payload twice with `PROFILE=0`: if those two runs also differ, the GPU run itself is not deterministic, and `scripts\compare_videos.py` (SSIM) is the right check.

The GPU sync at every mark makes each window slightly slower than without profiling. Compare the
`[SPEED] window` lines of both runs to see by how much.
