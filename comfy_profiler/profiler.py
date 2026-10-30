"""Per-window timing of the InfiniteTalk loop inside the ComfyUI process (PROFILE=1 only).

The whole loop runs inside ONE node (128 WanVideoSampler), so websocket messages cannot show
where the time between two windows goes. On the first queued prompt this module wraps,
without changing any argument or result:

  - nodes_sampler.multitalk_loop                 loop start / end, then the report
  - multitalk_loop module globals tqdm, mm, gc   step loop start, every step, empty_cache, gc.collect
  - WanVideoVAE.encode / decode / to             VAE time and frame counts

Only the wrapper's own references are replaced (module globals and the VAE class), nothing
in ComfyUI, torch or tqdm. Every mark waits for the GPU (torch.cuda.synchronize) so the times
are GPU time, not kernel launch time; this adds a little time, it does not change numerics.

Window k runs from the end of window k-1 (its step progress bar closing, right after the
decode) to the end of window k. So "prep" (window start -> first VAE call) holds the previous
window's post-processing (frame slicing, audio padding) plus this window's audio slice and
noise; in window 1 it holds the loop setup instead. The final torch.cat and the end-of-sampler
offload are reported as "after last window".
"""
import functools
import json
import logging
import os
import time

logger = logging.getLogger("worker_profiler")

WRAPPER_DIR = "ComfyUI-WanVideoWrapper"
TARGET_FILES = {
    "sampler": "nodes_sampler.py",
    "loop": "multitalk/multitalk_loop.py",
    "vae": "wanvideo/wan_video_vae.py",
}
STEP_BAR_DESC = "Sampling audio indices"
VAE_KINDS = ("vae_encode", "vae_decode", "vae_move")

# Swappable in tests
_clock = time.perf_counter
_cuda_sync = None

_active = None      # LoopRecorder of the loop that is running
_installed = None   # None: not tried yet, True: patched, False: disabled
_internal_error_logged = False


def _sync():
    if _cuda_sync is None:
        return
    try:
        _cuda_sync()
    except Exception:
        pass


def _safe(fn):
    """Runs profiler bookkeeping; a profiler bug must never break a job."""
    global _internal_error_logged
    try:
        return fn()
    except Exception as e:
        if not _internal_error_logged:
            _internal_error_logged = True
            logger.warning(f"[PROFILE] internal error (job unaffected): {type(e).__name__}: {e}")
        return None


class Window:
    def __init__(self, index, start):
        self.index = index
        self.start = start
        self.end = None
        self.steps_start = None
        self.step_marks = []
        self.events = []  # (kind, t0, t1, frames)

    def summary(self):
        end = self.end
        steps_end = None
        if self.steps_start is not None:
            last = self.step_marks[-1] if self.step_marks else self.steps_start
            # The steps end where the VAE work for the decode starts (after the last scheduler update)
            after = [e[1] for e in self.events if e[0] in VAE_KINDS and e[1] >= last]
            steps_end = min(after) if after else end
            steps = steps_end - self.steps_start
        else:
            steps = 0.0

        def outside(e):
            return steps_end is None or e[2] <= self.steps_start or e[1] >= steps_end

        def total(kind):
            return sum(e[2] - e[1] for e in self.events if e[0] == kind and outside(e))

        enc = [(e[3], e[2] - e[1]) for e in self.events if e[0] == "vae_encode" and outside(e)]
        dec = [(e[3], e[2] - e[1]) for e in self.events if e[0] == "vae_decode" and outside(e)]
        moves = [e for e in self.events if e[0] == "vae_move" and outside(e)]
        first_event = min([e[1] for e in self.events] + ([self.steps_start] if self.steps_start is not None else []),
                          default=end)
        window_total = end - self.start
        vae_encode, vae_decode = total("vae_encode"), total("vae_decode")
        vae_move, cache = total("vae_move"), total("empty_cache") + total("gc")
        other = window_total - steps - vae_encode - vae_decode
        prep = max(0.0, first_event - self.start)
        marks = self.step_marks
        return {
            "window": self.index,
            "total": window_total,
            "steps": steps,
            "vae_decode": vae_decode,
            "vae_encode": vae_encode,
            "other": other,
            "encode_calls": enc,
            "decode_calls": dec,
            "vae_move": vae_move,
            "vae_move_calls": len(moves),
            "empty_cache_gc": cache,
            "prep": prep,
            "untracked": other - vae_move - cache - prep,
            "step_count": len(marks),
            "first_step": (marks[0] - self.steps_start) if marks and self.steps_start is not None else None,
            "avg_s_per_it": (marks[-1] - marks[0]) / (len(marks) - 1) if len(marks) > 1 else None,
        }


class LoopRecorder:
    def __init__(self, t):
        self.begin = t
        self.windows = []
        self.current = Window(1, t)

    def event(self, kind, t0, t1, frames=None):
        self.current.events.append((kind, t0, t1, frames))

    def steps_start(self, t):
        self.current.steps_start = t

    def step(self, t):
        self.current.step_marks.append(t)

    def window_end(self, t):
        w = self.current
        w.end = t
        self.windows.append(w)
        self.current = Window(w.index + 1, t)


def _fmt_calls(calls, unit):
    if not calls:
        return "none"
    return " + ".join(f"{frames if frames is not None else '?'}{unit} {sec:.2f}s" for frames, sec in calls)


def _fmt_opt(value, spec, suffix=""):
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def report(rec, end, ok=True):
    """Logs the per-window lines and the summary; returns the data (also written as JSON)."""
    rows = [w.summary() for w in rec.windows]
    n = len(rows)
    loop_total = end - rec.begin
    tail = end - (rec.windows[-1].end if rec.windows else rec.begin)
    status = "" if ok else " (loop raised an exception; windows finished so far)"
    for r in rows:
        logger.info(
            f"[PROFILE] window {r['window']}/{n}: steps {r['steps']:.1f}s, vae_decode {r['vae_decode']:.1f}s, "
            f"vae_encode {r['vae_encode']:.1f}s, other {r['other']:.1f}s"
        )
        logger.info(
            f"[PROFILE] window {r['window']}/{n} detail: total {r['total']:.1f}s; "
            f"encode {_fmt_calls(r['encode_calls'], 'f')}; decode {_fmt_calls(r['decode_calls'], ' latent f')}; "
            f"other = vae_move {r['vae_move']:.2f}s ({r['vae_move_calls']}x), empty_cache+gc {r['empty_cache_gc']:.2f}s, "
            f"prep {r['prep']:.2f}s, untracked {r['untracked']:.2f}s; "
            f"first_step {_fmt_opt(r['first_step'], '.2f', 's')}, avg {_fmt_opt(r['avg_s_per_it'], '.2f', ' s/it')} "
            f"({r['step_count']} steps)"
        )
    keys = ("steps", "vae_decode", "vae_encode", "other")
    sums = {k: sum(r[k] for r in rows) for k in keys}
    windowed = sum(r["total"] for r in rows)
    shares = ", ".join(
        f"{k} {sums[k]:.1f}s ({100 * sums[k] / windowed:.0f}%)" if windowed > 0 else f"{k} {sums[k]:.1f}s" for k in keys
    )
    logger.info(f"[PROFILE] loop: {n} windows in {loop_total:.1f}s{status}; {shares}; after last window {tail:.1f}s")
    later = rows[1:]
    if later:
        avg = {k: sum(r[k] for r in later) / len(later) for k in keys + ("total", "vae_move", "empty_cache_gc")}
        logger.info(
            f"[PROFILE] windows 2..{n} average: total {avg['total']:.1f}s, steps {avg['steps']:.1f}s, "
            f"vae_decode {avg['vae_decode']:.1f}s, vae_encode {avg['vae_encode']:.1f}s, other {avg['other']:.1f}s "
            f"(vae_move {avg['vae_move']:.2f}s, empty_cache+gc {avg['empty_cache_gc']:.2f}s)"
        )
    data = {"ok": ok, "loop_total": loop_total, "after_last_window": tail, "windows": rows}
    _write_json(data)
    return data


def _write_json(data):
    out_dir = os.environ.get("PROFILE_DIR", "/tmp/worker_profile")
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"loop_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        logger.info(f"[PROFILE] written {path}")
    except Exception as e:
        logger.warning(f"[PROFILE] could not write the JSON report: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- wrappers
def _frames(args, index=1):
    """Frame count (dim 2 of a [B, C, T, H, W] tensor) of positional argument `index`."""
    try:
        return int(args[index].shape[2])
    except Exception:
        return None


def _timed(kind, orig, with_frames=False):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        rec = _active
        if rec is None:
            return orig(*args, **kwargs)
        _sync()
        t0 = _clock()
        out = orig(*args, **kwargs)
        _sync()
        t1 = _clock()
        _safe(lambda: rec.event(kind, t0, t1, _frames(args) if with_frames else None))
        return out
    return wrapper


def _wrap_loop(orig):
    @functools.wraps(orig)
    def timed_multitalk_loop(*args, **kwargs):
        global _active
        if _active is not None:
            return orig(*args, **kwargs)
        _sync()
        rec = LoopRecorder(_clock())
        _active = rec
        ok = False
        try:
            out = orig(*args, **kwargs)
            ok = True
            return out
        finally:
            _active = None
            _sync()
            end = _clock()
            _safe(lambda: report(rec, end, ok))
    return timed_multitalk_loop


class _StepBar:
    """The loop's per-window step progress bar: update() = one step done, close() = window done."""

    def __init__(self, bar, rec):
        self._bar = bar
        self._rec = rec

    def update(self, *args, **kwargs):
        out = self._bar.update(*args, **kwargs)
        _sync()
        t = _clock()
        _safe(lambda: self._rec.step(t))
        return out

    def close(self, *args, **kwargs):
        out = self._bar.close(*args, **kwargs)
        _sync()
        t = _clock()
        _safe(lambda: self._rec.window_end(t))
        return out

    def __getattr__(self, name):
        return getattr(self._bar, name)


def _wrap_tqdm(orig):
    # No functools.wraps: tqdm is a class, wraps would copy its whole __dict__ onto the function
    def tqdm(*args, **kwargs):
        bar = orig(*args, **kwargs)
        rec = _active
        if rec is None or not str(kwargs.get("desc", "")).startswith(STEP_BAR_DESC):
            return bar
        _sync()
        t = _clock()
        _safe(lambda: rec.steps_start(t))
        return _StepBar(bar, rec)
    return tqdm


class _ModuleProxy:
    """Stands in for a module in ONE module's globals: every attribute is the real one,
    except the few wrapped functions."""

    def __init__(self, module, overrides):
        self._module = module
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._module, name)


# ---------------------------------------------------------------- install
def _norm(path):
    return str(path or "").replace("\\", "/")


def find_targets(modules, wrapper_dir=WRAPPER_DIR):
    """Returns ({"sampler": mod, "loop": mod, "vae": mod}, problems)."""
    found, problems = {}, []
    for key, rel in TARGET_FILES.items():
        suffix = f"/{wrapper_dir}/{rel}"
        hits = [m for m in list(modules.values()) if _norm(getattr(m, "__file__", None)).endswith(suffix)]
        hits = list({id(m): m for m in hits}.values())
        if len(hits) != 1:
            problems.append(f"{len(hits)} loaded module(s) for {wrapper_dir}/{rel}")
        else:
            found[key] = hits[0]
    if problems:
        return found, problems
    loop_fn = getattr(found["loop"], "multitalk_loop", None)
    if not callable(loop_fn):
        problems.append("multitalk_loop.multitalk_loop missing")
    elif getattr(found["sampler"], "multitalk_loop", None) is not loop_fn:
        problems.append("nodes_sampler.multitalk_loop is not multitalk_loop.multitalk_loop")
    for name in ("tqdm", "mm", "gc"):
        if not hasattr(found["loop"], name):
            problems.append(f"multitalk_loop module has no global '{name}'")
    if hasattr(found["loop"], "mm") and not callable(getattr(found["loop"].mm, "soft_empty_cache", None)):
        problems.append("mm.soft_empty_cache missing")
    vae_cls = getattr(found["vae"], "WanVideoVAE", None)
    if vae_cls is None or not all(callable(getattr(vae_cls, n, None)) for n in ("encode", "decode", "to")):
        problems.append("WanVideoVAE.encode/decode/to missing")
    return found, problems


def install(modules=None, wrapper_dir=WRAPPER_DIR):
    """Patches once. Returns True when the timers are active."""
    global _installed, _cuda_sync
    if _installed is not None:
        return _installed
    import sys
    targets, problems = find_targets(sys.modules if modules is None else modules, wrapper_dir)
    if problems:
        _installed = False
        logger.warning(f"[PROFILE] disabled, wrapper layout not as expected: {'; '.join(problems)}")
        return False
    if _cuda_sync is None:
        try:
            import torch
            if torch.cuda.is_available():
                _cuda_sync = torch.cuda.synchronize
        except Exception:
            pass
    sampler, loop, vae = targets["sampler"], targets["loop"], targets["vae"]
    sampler.multitalk_loop = _wrap_loop(sampler.multitalk_loop)
    loop.tqdm = _wrap_tqdm(loop.tqdm)
    loop.mm = _ModuleProxy(loop.mm, {"soft_empty_cache": _timed("empty_cache", loop.mm.soft_empty_cache)})
    loop.gc = _ModuleProxy(loop.gc, {"collect": _timed("gc", loop.gc.collect)})
    cls = vae.WanVideoVAE
    cls.encode = _timed("vae_encode", cls.encode, with_frames=True)
    cls.decode = _timed("vae_decode", cls.decode, with_frames=True)
    cls.to = _timed("vae_move", cls.to)
    _installed = True
    logger.info(
        "[PROFILE] enabled: timing multitalk_loop windows (steps, VAE encode/decode/move, empty_cache, gc); "
        f"GPU sync at every mark (cuda sync {'on' if _cuda_sync else 'off'})"
    )
    return True


def _on_prompt(json_data):
    _safe(install)
    return json_data


def register():
    """Called by the package __init__ when PROFILE=1: patches on the first queued prompt,
    when every custom node package has been imported."""
    from server import PromptServer
    PromptServer.instance.add_on_prompt_handler(_on_prompt)
    logger.info("[PROFILE] PROFILE=1: InfiniteTalk loop timers install on the first prompt")
