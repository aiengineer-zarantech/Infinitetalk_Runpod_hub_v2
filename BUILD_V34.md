# v34: build, model cache, test plan

Branch `perf/5090-speedup`. Build `Dockerfile.v34` (MODEL_SOURCE=hf_cache). **Do not build `Dockerfile.v34.baked`.**
Nothing here is pushed or deployed until the owner says so.

## 1. Build host

`docker build` needs a Docker daemon. A RunPod **CPU pod is itself a container and has no Docker daemon**, so
`docker build` does not work there. Use a VM instead: EC2 / GCP / Hetzner, Ubuntu 22.04+, x86_64, **100 GB+ disk**,
2+ vCPU. No GPU needed (the build installs nothing and runs no CUDA code).

Disk: v32 base ~34 GB pulled + v34 layers (a few MB) + headroom. 100 GB is enough. v34 adds no models.

## 2. Build, tag, push (private repo)

```bash
# on the VM
sudo apt-get update && sudo apt-get install -y docker.io git
sudo usermod -aG docker $USER && newgrp docker

# perf/5090-speedup is NOT on GitHub yet. Either push it (owner's go-ahead), then:
git clone -b perf/5090-speedup https://github.com/aiengineer-zarantech/Infinitetalk_Runpod_hub_v2.git
# or, with no push, copy a bundle from the dev machine:
#   (dev)  git bundle create v34.bundle perf/5090-speedup && scp v34.bundle <vm>:
#   (vm)   git clone -b perf/5090-speedup v34.bundle Infinitetalk_Runpod_hub_v2
cd Infinitetalk_Runpod_hub_v2
git log --oneline -1                      # expect the v34 commit

# Docker Hub: create a PRIVATE repo first (hub.docker.com > Repositories > Create, Visibility: Private),
# e.g. digamberjhazaran/infinitetalk-runpod-private. Log in with an access token (Account settings >
# Personal access tokens, scope Read & Write), never the password:
docker login -u digamberjhazaran           # paste the token at the prompt

IMG=digamberjhazaran/infinitetalk-runpod-private
docker build --platform linux/amd64 -f Dockerfile.v34 -t $IMG:v34 . 2>&1 | tee build_v34.log
#   last line must be: "OK: v34 built. Only handler/setup_models/entrypoint + 4 worker_guard files changed..."
#   a fingerprint failure means the base is not v32: stop and report, do not edit the hashes

# smoke test without GPU: files and env inside the image
docker run --rm --entrypoint bash $IMG:v34 -c \
  'echo $MODEL_SOURCE $HF_CACHE_FALLBACK $UPSCALE_MODELS_AT_STARTUP; ls /worker_guard/vocals.py; \
   test ! -e /test_worker_diag.py && echo "dev scripts removed"'

# push ONLY after the owner's go-ahead
docker push $IMG:v34
docker inspect --format '{{index .RepoDigests 0}}' $IMG:v34   # record this digest in TESTING.md
```

Push is ~a few MB of new layers if the private repo already holds the v32 layers; otherwise the full ~34 GB once.

RunPod needs credentials for a private image: RunPod console > Settings > **Container Registry Auth** >
add Docker Hub user + read-only access token, then select it in the test endpoint's template.

## 3. Private Hugging Face repo (one cached model per endpoint)

RunPod allows **one cached model per endpoint**, so all files go into one private repo, with this layout
(setup_models.py looks for `<snapshot>/<folder>/<file>`):

| Path in repo | Bytes |
|---|---|
| `diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` | 16,993,877,896 |
| `diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors` | 2,713,548,210 |
| `diffusion_models/MelBandRoformer_fp16.safetensors` | 456,479,072 |
| `diffusion_models/wav2vec2-chinese-base_fp16.safetensors` | 190,115,368 |
| `text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors` | 6,731,333,792 |
| `clip_vision/clip_vision_h.safetensors` | 1,264,219,396 |
| `vae/Wan2_1_VAE_bf16.safetensors` | 253,806,278 |
| `loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | 738,005,744 |
| optional 9th: `diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors` | 2,712,729,090 |

Total ~29.3 GB (32 GB with Multi). Without the 9th file, multi-person jobs need the network volume.

Upload from a RunPod CPU pod (no Docker needed for this) with the network volume attached, same datacenter:

```bash
pip install -U "huggingface_hub[hf_xet]"
hf auth login                                   # WRITE token, typed at the prompt
bash scripts/upload_hf_cache.sh <org>/infinitetalk-models /runpod-volume
# optional 9th file:
hf upload <org>/infinitetalk-models \
  "$(find /runpod-volume -name Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors | head -1)" \
  diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors
```

The script checks every size against the public source first and stops on a mismatch.

### Endpoint settings (test endpoint)

- **Model** field: `<org>/infinitetalk-models` (RunPod's documented format is `org/name`).
- **Hugging Face token**: a **read** token that can see the private repo (entered in the same Model section).
- Env: `MODEL_SOURCE=hf_cache` is already the image default. `HF_CACHE_REPO` can stay unset (the only
  cached repo is used) or be set to `<org>/infinitetalk-models`. Keep `HF_CACHE_FALLBACK=1`.
- Keep the network volume attached for the first test (fallback + compile cache). The RunPod docs do not say
  how a cached model and a network volume behave together on `/runpod-volume`; check the worker log line
  `[MODELS] MODEL_SOURCE=hf_cache ...` and `[HF_CACHE] ... linked` on the first start.
- Not verified from the docs: whether the cache is read-only and whether repo subfolders are kept as-is.
  The `[HF_CACHE]` lines show both on the first start.

## 4. What fp8_fast changes

- Today: weights are stored in fp8 (`fp8_e4m3fn`) but each matmul upcasts them to bf16.
  `fp8_e4m3fn_fast` runs the linear layers as real fp8 matmuls on the 5090 tensor cores (inputs cast to fp8 too).
- WanVideoWrapper's fast fp8 path cannot add an unmerged LoRA at runtime, hence the R2 failure
  `fp8_fast is not supported with unmerged LoRAs`. v34 sets `merge_loras=true` on the lightx2v LoRA node for
  fp8_fast jobs only: the LoRA delta is added into the base weights once at model load.
- Quality risk (to be measured, not assumed):
  1. Activations in fp8: small numeric error in every linear layer.
  2. The merged LoRA is stored at fp8 precision (3 mantissa bits). lightx2v is the step-distill LoRA that makes
     6 steps work, so rounding there can show as softer detail, colour shift or mouth artifacts.
  3. Model load gets slower (merge at load); it shows in `stages_sec.model_load`.
  How exactly the wrapper merges into fp8 weights (upcast first or not) is not verified here.
- fp16_fast (R3): fp16 compute with fp16 accumulation. Faster matmuls, but fp16 has a much smaller range than
  bf16, so overflow can give NaN / black or frozen frames. The v32 handler deliberately turned `fp16_fast` in
  the workflow into `bf16`; a per-job `base_precision` overrides that.

## 5. Timing in v34

- `[SPEED] window x/N: <s>, <s/it>` per window (already in v32; window 1 includes model move + compile).
- New: `[TIMING] stages: vocal_separation, wav2vec, t5, clip, model_load, image_embeds, sampler, vae_decode,
  video_encode, other` on every video job. Per-node `[PROFILE]` lines stay `PROFILE=1` only.
- Benchmark jobs return `output.timing`: `windows[]`, `avg_s_per_it`, `first_step_overhead_sec`, `sampling_sec`,
  `stages_sec`, `comfy_total_sec`, `fp8_fast`. Note: the sampler stage includes the per-window VAE decode/encode
  that InfiniteTalk does inside node 128; `s_per_it` isolates the diffusion steps.

## 6. Test plan (v34 TEST endpoint, after approval)

Backend repo, `diagnostics/speedtest/`:

```bash
TEST_ENDPOINT_ID=<v34 test endpoint id> python speedtest_v34.py ../../backend <test_dir>
python analyze.py <test_dir>/out R1 R2 R3
```

| Job | Input on top of the production payload | Purpose |
|---|---|---|
| W0 | benchmark | warm-up (cold start + compile), also checks hf_cache start |
| R1 | benchmark | warm baseline |
| R2 | benchmark, fp8_fast | fp8 matmuls + merged LoRA |
| R3 | benchmark, fp8_fast, base_precision fp16_fast | + fp16 accumulation |

- Same 10 s audio and image. The script refuses to run on the production endpoint.
- Pass criteria per run: COMPLETED, 250 frames, A/V diff <= 40 ms, no black/freeze, SSIM vs R1 reported
  (R0 vs R1 on v32 was 1.0, so any drop comes from the setting). Contact sheets for a visual check of the mouth.
- Cost: 4 jobs, about 25-30 min of 5090 time.
