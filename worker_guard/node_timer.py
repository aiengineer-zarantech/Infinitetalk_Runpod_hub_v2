"""Wall time per ComfyUI node from websocket "executing" messages (PROFILE=1 only).

A node runs from its "executing" message until the next one ("executing" with node None ends
the prompt). Covers the one-time nodes outside the sampler loop: audio embedding (194),
CLIP vision (237), text encoding, model loaders, video combine. The per-window split inside
node 128 comes from the ComfyUI-side profiler (comfy_profiler).
"""
import logging

logger = logging.getLogger("worker_guard")


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

    def log(self, node_classes):
        for node, sec in self.times:
            logger.info(f"[PROFILE] node {node} ({node_classes.get(node, '?')}): {sec:.1f}s")
        if self.current is not None:
            logger.info(f"[PROFILE] node {self.current} ({node_classes.get(self.current, '?')}): did not finish")
        total = sum(sec for _, sec in self.times)
        cached = f"; cached: {', '.join(self.cached)}" if self.cached else ""
        logger.info(f"[PROFILE] nodes: {len(self.times)} ran in {total:.1f}s{cached}")
