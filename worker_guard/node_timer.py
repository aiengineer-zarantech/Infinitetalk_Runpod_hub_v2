"""Wall time per ComfyUI node from websocket "executing" messages (v34: timed on every video job; per-node
[PROFILE] lines only with PROFILE=1, one [TIMING] stages line always).

A node runs from its "executing" message until the next one ("executing" with node None ends
the prompt). Covers the one-time nodes outside the sampler loop: audio embedding (194),
CLIP vision (237), text encoding, model loaders, video combine. The per-window split inside
node 128 comes from the ComfyUI-side profiler (comfy_profiler).
"""
import logging

logger = logging.getLogger("worker_guard")

# v34: node class -> stage. "sampler" is node 128 as a whole: every window's steps plus the per-window VAE
# decode/encode the InfiniteTalk loop does inside it ([SPEED] window lines split it per window).
STAGES = {
    "LoadWanVideoT5TextEncoder": "t5", "WanVideoTextEncode": "t5", "WanVideoTextEncodeCached": "t5",
    "WanVideoApplyNAG": "t5",
    "CLIPVisionLoader": "clip", "WanVideoClipVisionEncode": "clip",
    "Wav2VecModelLoader": "wav2vec", "DownloadAndLoadWav2VecModel": "wav2vec", "MultiTalkWav2VecEmbeds": "wav2vec",
    "MelBandRoFormerModelLoader": "vocal_separation", "MelBandRoFormerSampler": "vocal_separation",
    "WanVideoModelLoader": "model_load", "WanVideoLoraSelect": "model_load", "MultiTalkModelLoader": "model_load",
    "WanVideoBlockSwap": "model_load", "WanVideoTorchCompileSettings": "model_load", "WanVideoVAELoader": "model_load",
    "WanVideoImageToVideoMultiTalk": "image_embeds", "WanVideoEncode": "image_embeds",
    "WanVideoSampler": "sampler",
    "WanVideoDecode": "vae_decode",
    "VHS_VideoCombine": "video_encode",
}
STAGE_ORDER = ("vocal_separation", "wav2vec", "t5", "clip", "model_load", "image_embeds", "sampler",
               "vae_decode", "video_encode", "other")


class NodeTimer:
    def __init__(self):
        self.current = None
        self.started = None
        self.times = []  # (node, seconds) in execution order
        self.cached = []

    def on_executing(self, node, t):
        if self.current is not None:
            self.times.append((self.current, t - self.started))
        if node is None:
            self.current = self.started = None
        else:
            self.current, self.started = str(node), t

    def on_cached(self, nodes):
        self.cached.extend(str(n) for n in nodes or [])

    def stages(self, node_classes):
        """Seconds per stage (only stages that ran), in STAGE_ORDER."""
        out = {}
        for node, sec in self.times:
            stage = STAGES.get(node_classes.get(node, ""), "other")
            out[stage] = out.get(stage, 0.0) + sec
        return {k: round(out[k], 2) for k in STAGE_ORDER if k in out}

    def log(self, node_classes):
        for node, sec in self.times:
            logger.info(f"[PROFILE] node {node} ({node_classes.get(node, '?')}): {sec:.1f}s")
        if self.current is not None:
            logger.info(f"[PROFILE] node {self.current} ({node_classes.get(self.current, '?')}): did not finish")
        total = sum(sec for _, sec in self.times)
        cached = f"; cached: {', '.join(self.cached)}" if self.cached else ""
        logger.info(f"[PROFILE] nodes: {len(self.times)} ran in {total:.1f}s{cached}")

    def log_stages(self, node_classes):
        """v34: one line on every video job (the per-node [PROFILE] lines stay PROFILE=1 only)."""
        stages = self.stages(node_classes)
        if stages:
            logger.info("[TIMING] stages: " + ", ".join(f"{k} {v:.1f}s" for k, v in stages.items()))
