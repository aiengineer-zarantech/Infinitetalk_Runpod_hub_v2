"""Sampler speed from ComfyUI websocket messages.

The InfiniteTalk loop (multitalk_loop.py) reports ONE progress counter for all windows:
value = global step (1-based), max = windows * steps_per_window. Each window ends with its
VAE decode inside the same sampler node, before the next window's first step.
"""
import logging

logger = logging.getLogger("worker_guard")


class SpeedTracker:
    def __init__(self, steps_per_window, sampler_node="128"):
        self.spw = steps_per_window
        self.sampler_node = str(sampler_node)
        self.start = None
        self.end = None
        self.first_step_at = None
        self.step_times = {}
        self.windows = []  # (index, window_sec, s_per_it)
        self.last_window_end = None
        self.total = None
        self._warned = False

    def on_executing(self, node, t):
        if node is not None and str(node) == self.sampler_node:
            if self.start is None:
                self.start = self.last_window_end = t
        elif self.start is not None and self.end is None:
            self.end = t

    def on_progress(self, node, value, total, t):
        if node is not None and str(node) != self.sampler_node:
            return
        try:
            value, total = int(value), int(total)
        except (TypeError, ValueError):
            return
        if self.start is None:
            self.start = self.last_window_end = t
        spw = self.spw
        if total != self.total:
            # Node 128 first reports other progress (e.g. "Loading transformer parameters", max 1633);
            # a new max starts a new phase, and nothing measured in the previous phase is kept.
            if self.windows:
                logger.warning(f"[SPEED] progress max changed {self.total} -> {total}; discarding {len(self.windows)} window(s)")
            self.total = total
            self.step_times, self.windows = {}, []
            self.first_step_at = None
            self.last_window_end = self.start
        if not spw or total % spw:
            if not self._warned:
                self._warned = True
                logger.warning(f"[SPEED] ignoring node {self.sampler_node} progress with max {total} "
                               f"(not windows x {spw} steps, e.g. model loading)")
            return
        if self.first_step_at is None:
            self.first_step_at = t
        self.step_times[value] = t
        if value % spw:
            return
        index, windows = value // spw, total // spw
        window_sec = t - self.last_window_end
        self.last_window_end = t
        first = self.step_times.get(value - spw + 1)
        if spw > 1 and first is not None:
            s_per_it = (t - first) / (spw - 1)
        else:
            s_per_it = window_sec / spw
        self.windows.append((index, window_sec, s_per_it))
        logger.info(f"[SPEED] window {index}/{windows}: {window_sec:.1f}s, {s_per_it:.2f} s/it")

    def avg_s_per_it(self):
        if not self.windows:
            return None
        return sum(w[2] for w in self.windows) / len(self.windows)

    def sampling_sec(self):
        if self.start is None:
            return None
        end = self.end if self.end is not None else self.last_window_end
        return None if end is None else end - self.start

    def first_step_overhead(self):
        """Delay before the first step, minus one normal step: compile + moving the model to the GPU."""
        avg = self.avg_s_per_it()
        if self.start is None or self.first_step_at is None or avg is None:
            return None
        return max(0.0, (self.first_step_at - self.start) - avg)
