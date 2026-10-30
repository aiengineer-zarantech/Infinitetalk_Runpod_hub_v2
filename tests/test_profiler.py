# ==============================================================================
# comfy_profiler (PROFILE=1 per-window timing inside ComfyUI), worker_guard/node_timer.py,
# the job_guard PROFILE hook and scripts/verify_profiler_contract.py. No torch, no GPU:
# a fake wrapper replays the call order of the real multitalk_loop with a fake clock.
# ==============================================================================
import importlib
import json
import logging
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

from comfy_profiler import profiler
from worker_guard import job_guard
from worker_guard.node_timer import NodeTimer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import verify_profiler_contract  # noqa: E402

WRAPPER = "/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"

# Same order of calls as multitalk_loop.py (088128b) for mode "infinitetalk", no offload, no tiling
FAKE_LOOP_SRC = '''
def multitalk_loop(self=None, vae=None, windows=3, steps=6, clock=None, cost=None, fail_decode_in=None):
    clock.advance(cost["setup"])
    out = []
    for w in range(windows):
        clock.advance(cost["prep"])
        vae.to("cuda")
        y = vae.encode(Frames(81), "cuda")
        motion = vae.encode(Frames(1 if w == 0 else 9), "cuda")
        vae.to("cpu")
        mm.soft_empty_cache()
        clock.advance(cost["sched"])
        mm.soft_empty_cache()
        gc.collect()
        bar = tqdm(total=steps, desc=f"Sampling audio indices {w}-{w + 81}", position=0, leave=True)
        for i in range(steps):
            clock.advance(cost["step"])
            bar.update(1)
        clock.advance(cost["sched_last"])
        vae.to("cuda")
        if fail_decode_in == w + 1:
            raise RuntimeError("CUDA out of memory (fake)")
        videos = vae.decode(Frames(21), "cuda")
        clock.advance(cost["cpu_copy"])
        vae.to("cpu")
        bar.close()
        clock.advance(cost["post"])
        out.append((y, motion, videos, mm.get_torch_device()))
    clock.advance(cost["tail"])
    return out
'''

COST = {"setup": 2.0, "prep": 0.5, "move": 0.1, "enc_per_frame": 0.05, "cache": 0.2, "gc": 0.3, "sched": 0.05,
        "step": 9.0, "sched_last": 0.02, "decode": 7.0, "cpu_copy": 0.08, "post": 0.3, "tail": 1.0}


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class Frames:
    def __init__(self, t):
        self.shape = (1, 3, t, 480, 720)


class FakeBar:
    def __init__(self, total=None, desc="", **kwargs):
        self.total, self.desc, self.n, self.closed = total, desc, 0, False

    def update(self, n=1):
        self.n += n

    def close(self):
        self.closed = True


def make_wrapper(clock, cost=COST):
    """Fresh fake wrapper modules (sampler, loop, vae) as ComfyUI would have them in sys.modules."""
    vae_mod = types.ModuleType(f"{WRAPPER}.wanvideo.wan_video_vae")
    vae_mod.__file__ = f"{WRAPPER}/wanvideo/wan_video_vae.py"

    class Module:
        def to(self, *args, **kwargs):
            clock.advance(cost["move"])
            return self

    class WanVideoVAE(Module):
        def encode(self, videos, device, tiled=False, pbar=True):
            clock.advance(cost["enc_per_frame"] * videos.shape[2])
            return ("latent", videos.shape[2])

        def decode(self, hidden_states, device, tiled=False, pbar=True):
            clock.advance(cost["decode"])
            return ["video", hidden_states.shape[2]]

    vae_mod.WanVideoVAE = WanVideoVAE

    loop_mod = types.ModuleType(f"{WRAPPER}.multitalk.multitalk_loop")
    loop_mod.__file__ = f"{WRAPPER}/multitalk/multitalk_loop.py"
    loop_mod.__dict__.update(
        Frames=Frames,
        tqdm=FakeBar,
        mm=types.SimpleNamespace(soft_empty_cache=lambda force=False: clock.advance(cost["cache"]),
                                 get_torch_device=lambda: "cuda:0"),
        gc=types.SimpleNamespace(collect=lambda: clock.advance(cost["gc"]) or 0),
    )
    exec(FAKE_LOOP_SRC, loop_mod.__dict__)

    sampler_mod = types.ModuleType(f"{WRAPPER}.nodes_sampler")
    sampler_mod.__file__ = f"{WRAPPER}/nodes_sampler.py"
    sampler_mod.multitalk_loop = loop_mod.multitalk_loop
    modules = {m.__name__: m for m in (vae_mod, loop_mod, sampler_mod)}
    modules["unrelated"] = types.ModuleType("unrelated")
    return types.SimpleNamespace(modules=modules, vae=vae_mod, loop=loop_mod, sampler=sampler_mod)


@pytest.fixture
def prof(monkeypatch, tmp_path):
    clock = FakeClock()
    syncs = []
    monkeypatch.setattr(profiler, "_clock", clock)
    monkeypatch.setattr(profiler, "_cuda_sync", lambda: syncs.append(1))
    monkeypatch.setattr(profiler, "_active", None)
    monkeypatch.setattr(profiler, "_installed", None)
    monkeypatch.setattr(profiler, "_internal_error_logged", False)
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path / "profile"))
    w = make_wrapper(clock)
    return types.SimpleNamespace(clock=clock, syncs=syncs, w=w, dir=tmp_path / "profile")


def run_loop(p, **kwargs):
    vae = p.w.vae.WanVideoVAE()
    return p.w.sampler.multitalk_loop(self=None, vae=vae, clock=p.clock, cost=COST, **kwargs)


# ---------------------------------------------------------------- window split
def test_window_split_matches_known_costs(prof, caplog):
    assert profiler.install(prof.w.modules) is True
    caplog.set_level(logging.INFO, logger="worker_profiler")
    run_loop(prof, windows=3)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[PROFILE] window")]

    # Window 2: previous post 0.3 + prep 0.5; 4 moves; encode 81f + 9f; 2 empty_cache + gc; 6 steps + last update;
    # decode; cpu copy. Steps end where vae.to() for the decode starts.
    assert "[PROFILE] window 2/3: steps 54.0s, vae_decode 7.0s, vae_encode 4.5s, other 2.0s" in lines
    assert "[PROFILE] window 1/3: steps 54.0s, vae_decode 7.0s, vae_encode 4.1s, other 3.7s" in lines
    detail = next(line for line in lines if line.startswith("[PROFILE] window 2/3 detail"))
    assert "total 67.5s" in detail or "total 67.6s" in detail
    assert "encode 81f 4.05s + 9f 0.45s" in detail
    assert "decode 21 latent f 7.00s" in detail
    assert "vae_move 0.40s (4x), empty_cache+gc 0.70s, prep 0.80s, untracked 0.13s" in detail
    assert "first_step 9.00s, avg 9.00 s/it (6 steps)" in detail
    w1 = next(line for line in lines if line.startswith("[PROFILE] window 1/3 detail"))
    assert "encode 81f 4.05s + 1f 0.05s" in w1 and "prep 2.50s" in w1

    data = json.loads(next(prof.dir.glob("loop_*.json")).read_text(encoding="utf-8"))
    w2 = data["windows"][1]
    assert w2["steps"] == pytest.approx(54.02)
    assert w2["vae_encode"] == pytest.approx(4.5)
    assert w2["vae_decode"] == pytest.approx(7.0)
    assert w2["total"] == pytest.approx(w2["steps"] + w2["vae_encode"] + w2["vae_decode"] + w2["other"])
    assert w2["untracked"] == pytest.approx(0.05 + 0.08)
    assert data["after_last_window"] == pytest.approx(0.3 + 1.0)
    assert data["ok"] is True

    summary = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("[PROFILE] loop:"))
    assert summary.startswith("[PROFILE] loop: 3 windows in ")
    assert "after last window 1.3s" in summary
    assert any(r.getMessage().startswith("[PROFILE] windows 2..3 average: total 67.5s") or
               r.getMessage().startswith("[PROFILE] windows 2..3 average: total 67.6s") for r in caplog.records)
    assert prof.syncs, "GPU must be synchronized at the marks"


def test_results_identical_to_unpatched(prof):
    clean = make_wrapper(FakeClock())
    expected = clean.sampler.multitalk_loop(vae=clean.vae.WanVideoVAE(), clock=FakeClock(), cost=COST, windows=2)
    profiler.install(prof.w.modules)
    assert run_loop(prof, windows=2) == expected


def test_vae_and_helpers_outside_loop_pass_through(prof, caplog):
    profiler.install(prof.w.modules)
    caplog.set_level(logging.INFO, logger="worker_profiler")
    vae = prof.w.vae.WanVideoVAE()
    assert vae.encode(Frames(5), "cuda") == ("latent", 5)
    assert vae.to("cpu") is vae
    assert prof.w.loop.mm.get_torch_device() == "cuda:0"
    prof.w.loop.gc.collect()
    bar = prof.w.loop.tqdm(total=3, desc="Sampling audio indices 0-81")
    assert isinstance(bar, FakeBar)  # no loop running: the real bar, unwrapped
    assert prof.syncs == []
    assert not any(r.getMessage().startswith("[PROFILE] window") for r in caplog.records)


@pytest.mark.parametrize("call", [
    lambda w, vae: vae.encode(Frames(81), "cuda"),
    lambda w, vae: vae.decode(Frames(21), "cuda"),
    lambda w, vae: vae.to("cuda"),
    lambda w, vae: w.loop.mm.soft_empty_cache(),
    lambda w, vae: w.loop.gc.collect(),
])
def test_every_timed_call_waits_for_the_gpu_before_and_after(prof, call):
    profiler.install(prof.w.modules)
    vae = prof.w.vae.WanVideoVAE()
    profiler._active = rec = profiler.LoopRecorder(prof.clock())
    try:
        start = prof.clock()
        call(prof.w, vae)
    finally:
        profiler._active = None
    assert len(prof.syncs) == 2  # queued GPU work must not leak into (or out of) the measured span
    (kind, t0, t1, _), = rec.current.events
    assert t0 == start and t1 > t0


def test_other_progress_bars_are_not_wrapped(prof):
    profiler.install(prof.w.modules)
    profiler._active = profiler.LoopRecorder(0.0)
    try:
        bar = prof.w.loop.tqdm(total=3, desc="Loading transformer parameters")
    finally:
        profiler._active = None
    assert isinstance(bar, FakeBar)


def test_exception_propagates_and_partial_report(prof, caplog):
    profiler.install(prof.w.modules)
    caplog.set_level(logging.INFO, logger="worker_profiler")
    with pytest.raises(RuntimeError, match="out of memory"):
        run_loop(prof, windows=3, fail_decode_in=2)
    assert profiler._active is None
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[PROFILE] window 1/1: ") for m in messages)
    assert any("loop raised an exception" in m for m in messages)
    # the next job is profiled normally
    caplog.clear()
    run_loop(prof, windows=2)
    assert any(r.getMessage().startswith("[PROFILE] window 2/2: ") for r in caplog.records)


def test_profiler_bug_never_breaks_the_job(prof, monkeypatch, caplog):
    profiler.install(prof.w.modules)
    clean = make_wrapper(FakeClock())
    expected = clean.sampler.multitalk_loop(vae=clean.vae.WanVideoVAE(), clock=FakeClock(), cost=COST, windows=2)

    def broken(*a, **k):
        raise ValueError("profiler bug")

    monkeypatch.setattr(profiler.LoopRecorder, "event", broken)
    monkeypatch.setattr(profiler, "report", broken)
    caplog.set_level(logging.WARNING, logger="worker_profiler")
    assert run_loop(prof, windows=2) == expected
    warnings = [r.getMessage() for r in caplog.records if "internal error" in r.getMessage()]
    assert len(warnings) == 1 and "job unaffected" in warnings[0]


def test_cuda_sync_error_is_ignored(prof, monkeypatch):
    profiler.install(prof.w.modules)

    def bad_sync():
        raise RuntimeError("sync failed")

    monkeypatch.setattr(profiler, "_cuda_sync", bad_sync)
    assert len(run_loop(prof, windows=1)) == 1


def test_install_is_idempotent(prof, caplog):
    assert profiler.install(prof.w.modules) is True
    first = prof.w.sampler.multitalk_loop
    assert profiler.install(prof.w.modules) is True
    assert prof.w.sampler.multitalk_loop is first
    caplog.set_level(logging.INFO, logger="worker_profiler")
    run_loop(prof, windows=2)
    assert sum(r.getMessage().startswith("[PROFILE] loop:") for r in caplog.records) == 1


@pytest.mark.parametrize("breakage, message", [
    (lambda w: w.modules.pop(w.loop.__name__), "0 loaded module(s) for ComfyUI-WanVideoWrapper/multitalk/multitalk_loop.py"),
    (lambda w: setattr(w.sampler, "multitalk_loop", lambda **k: None), "is not multitalk_loop.multitalk_loop"),
    (lambda w: delattr(w.loop, "gc"), "no global 'gc'"),
    (lambda w: delattr(w.vae.WanVideoVAE, "decode"), "WanVideoVAE.encode/decode/to missing"),
])
def test_unexpected_layout_disables_without_patching(prof, caplog, breakage, message):
    breakage(prof.w)
    before = dict(vars(prof.w.loop)), dict(vars(prof.w.sampler)), dict(vars(prof.w.vae.WanVideoVAE))
    caplog.set_level(logging.WARNING, logger="worker_profiler")
    assert profiler.install(prof.w.modules) is False
    assert (dict(vars(prof.w.loop)), dict(vars(prof.w.sampler)), dict(vars(prof.w.vae.WanVideoVAE))) == before
    assert any(message in r.getMessage() for r in caplog.records)
    assert profiler.install(prof.w.modules) is False  # tried once only


def test_windows_paths_are_matched(prof):
    for m in prof.w.modules.values():
        if getattr(m, "__file__", None):
            m.__file__ = "C:" + m.__file__.replace("/", "\\")
    assert profiler.install(prof.w.modules) is True


# ---------------------------------------------------------------- package gate
def _fresh_package(monkeypatch):
    for name in ("comfy_profiler", "comfy_profiler.profiler"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("comfy_profiler")


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_package_does_nothing_without_profile(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PROFILE", raising=False)
    else:
        monkeypatch.setenv("PROFILE", value)
    monkeypatch.setitem(sys.modules, "server", None)  # importing server would fail
    pkg = _fresh_package(monkeypatch)
    assert pkg.NODE_CLASS_MAPPINGS == {} and pkg.NODE_DISPLAY_NAME_MAPPINGS == {}
    assert "comfy_profiler.profiler" not in sys.modules


def test_package_registers_prompt_hook_with_profile(monkeypatch):
    handlers = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(add_on_prompt_handler=handlers.append))
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setenv("PROFILE", "1")
    pkg = _fresh_package(monkeypatch)
    assert pkg.NODE_CLASS_MAPPINGS == {}
    assert len(handlers) == 1
    prof_mod = sys.modules["comfy_profiler.profiler"]
    monkeypatch.setattr(prof_mod, "_installed", None)
    payload = {"prompt": {"1": {}}, "client_id": "x"}
    assert handlers[0](payload) is payload  # prompt passed through unchanged
    assert prof_mod._installed is False  # no wrapper loaded in tests: disabled, not crashed


def test_package_survives_missing_server(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "server", None)
    monkeypatch.setenv("PROFILE", "1")
    caplog.set_level(logging.WARNING, logger="worker_profiler")
    pkg = _fresh_package(monkeypatch)
    assert pkg.NODE_CLASS_MAPPINGS == {}
    assert any("[PROFILE] not enabled" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- handler side: node times
def test_node_timer_lines(caplog):
    t = NodeTimer()
    t.on_cached(["241"])
    for node, at in (("194", 0.0), ("237", 3.5), ("128", 5.0), ("131", 300.0), (None, 312.0)):
        t.on_executing(node, at)
    caplog.set_level(logging.INFO, logger="worker_guard")
    t.log({"194": "MultiTalkWav2VecEmbeds", "237": "WanVideoClipVisionEncode", "128": "WanVideoSampler",
           "131": "VHS_VideoCombine"})
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == [
        "[PROFILE] node 194 (MultiTalkWav2VecEmbeds): 3.5s",
        "[PROFILE] node 237 (WanVideoClipVisionEncode): 1.5s",
        "[PROFILE] node 128 (WanVideoSampler): 295.0s",
        "[PROFILE] node 131 (VHS_VideoCombine): 12.0s",
        "[PROFILE] nodes: 4 ran in 312.0s; cached: 241",
    ]


def test_node_timer_unfinished_node(caplog):
    t = NodeTimer()
    t.on_executing("128", 0.0)
    caplog.set_level(logging.INFO, logger="worker_guard")
    t.log({"128": "WanVideoSampler"})
    assert "[PROFILE] node 128 (WanVideoSampler): did not finish" in [r.getMessage() for r in caplog.records]


class ClockedWs:
    """Returns scripted messages; each recv() first moves the job_guard clock forward."""

    def __init__(self, clock, messages):
        self.clock, self.messages = clock, list(messages)

    def recv(self):
        at, message = self.messages.pop(0)
        self.clock["t"] = at
        return json.dumps(message)


def _run_job(monkeypatch, profile):
    clock = {"t": 1000.0}
    monkeypatch.setattr(job_guard, "_now", lambda: clock["t"])
    if profile is None:
        monkeypatch.delenv("PROFILE", raising=False)
    else:
        monkeypatch.setenv("PROFILE", profile)
    messages = [
        (1000.0, {"type": "execution_start", "data": {"prompt_id": "p1"}}),
        (1000.0, {"type": "execution_cached", "data": {"nodes": ["241"], "prompt_id": "p1"}}),
        (1001.0, {"type": "executing", "data": {"node": "194", "prompt_id": "p1"}}),
        (1004.0, {"type": "executing", "data": {"node": "128", "prompt_id": "p1"}}),
        (1300.0, {"type": "executing", "data": {"node": "131", "prompt_id": "p1"}}),
        (1310.0, {"type": "executing", "data": {"node": None, "prompt_id": "p1"}}),
    ]
    prompt = {"128": {"class_type": "WanVideoSampler", "inputs": {"steps": 6}},
              "131": {"class_type": "VHS_VideoCombine", "inputs": {}},
              "194": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {}}}

    def get_videos(ws, prompt, input_type="image", person_count="single"):
        for _ in range(len(messages)):
            ws.recv()
        return {"131": ["/out.mp4"]}

    holder = {}

    def inner(job):
        holder["out"] = gv(ClockedWs(clock, messages), prompt)
        return {"video": "b64"}

    monkeypatch.setenv("TORCH_COMPILE", "0")
    gh, gv = job_guard.install(inner, get_videos)
    result = gh({"input": {"width": 720, "height": 480}})
    return result, holder["out"]


def test_job_guard_logs_node_times_with_profile(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    result, videos = _run_job(monkeypatch, "1")
    assert result == {"video": "b64"} and videos == {"131": ["/out.mp4"]}
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[PROFILE]")]
    assert msgs == [
        "[PROFILE] node 194 (MultiTalkWav2VecEmbeds): 3.0s",
        "[PROFILE] node 128 (WanVideoSampler): 296.0s",
        "[PROFILE] node 131 (VHS_VideoCombine): 10.0s",
        "[PROFILE] nodes: 3 ran in 309.0s; cached: 241",
    ]


@pytest.mark.parametrize("profile", [None, "0"])
def test_job_guard_no_profile_lines_without_profile(monkeypatch, caplog, profile):
    caplog.set_level(logging.INFO, logger="worker_guard")
    result, videos = _run_job(monkeypatch, profile)
    assert result == {"video": "b64"} and videos == {"131": ["/out.mp4"]}
    assert not any(r.getMessage().startswith("[PROFILE]") for r in caplog.records)


# ---------------------------------------------------------------- build-time contract
MINI_SAMPLER = """
from .multitalk.multitalk_loop import multitalk_loop
class WanVideoSampler:
    def process(self, **kw):
        return multitalk_loop(**locals())
"""

MINI_LOOP = """
import os
import gc
import comfy.model_management as mm
from tqdm import tqdm

def multitalk_loop(self, **kwargs):
    while True:
        vae.to(device)
        y = vae.encode(padding, device=device, tiled=tiled_vae, pbar=False)
        vae.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()
        sampling_pbar = tqdm(total=len(timesteps)-1, desc=f"Sampling audio indices {a}-{b}", position=0, leave=True)
        for i in range(len(timesteps)-1):
            sampling_pbar.update(1)
        vae.to(device)
        videos = vae.decode(latent, device=device, tiled=tiled_vae, pbar=False)[0].cpu()
        vae.to(offload_device)
        sampling_pbar.close()
"""

MINI_VAE = """
class WanVideoVAE(nn.Module):
    def encode(self, videos, device, tiled=False):
        pass
    def decode(self, hidden_states, device, tiled=False):
        pass
"""


def mini_wrapper(tmp_path, loop=MINI_LOOP, vae=MINI_VAE, sampler=MINI_SAMPLER):
    root = tmp_path / "ComfyUI-WanVideoWrapper"
    for rel, src in (("nodes_sampler.py", sampler), ("multitalk/multitalk_loop.py", loop),
                     ("wanvideo/wan_video_vae.py", vae)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(root)


def test_contract_ok_on_matching_layout(tmp_path):
    assert verify_profiler_contract.check(mini_wrapper(tmp_path)) == []
    assert verify_profiler_contract.main([mini_wrapper(tmp_path)]) == 0


@pytest.mark.parametrize("part, old, new, message", [
    ("loop", 'desc=f"Sampling audio indices', 'desc=f"Denoising', "expected ONE"),
    ("loop", "import gc\n", "", "no 'import gc'"),
    ("loop", "from tqdm import tqdm", "from tqdm.auto import tqdm", "no 'from tqdm import tqdm'"),
    ("loop", "        sampling_pbar.close()\n", "", "decode() or the step bar close() missing"),
    ("loop", "            sampling_pbar.update(1)\n", "            pass\n", "expected one sampling_pbar.update()"),
    ("loop", "        gc.collect()\n", "", "gc.collect() is never called"),
    ("vae", "def encode(self, videos", "def encode(self, video_list", "WanVideoVAE.encode"),
    ("vae", "    def decode(self, hidden_states, device, tiled=False):\n        pass\n",
     "    def decode(self, hidden_states, device, tiled=False):\n        pass\n    def to(self, *a):\n        pass\n",
     "defines its own to()"),
    ("sampler", "multitalk_loop(**locals())", "other(**locals())", "never called by name"),
])
def test_contract_detects_changes(tmp_path, part, old, new, message):
    srcs = {"loop": MINI_LOOP, "vae": MINI_VAE, "sampler": MINI_SAMPLER}
    assert old in srcs[part]
    srcs[part] = srcs[part].replace(old, new)
    errors = verify_profiler_contract.check(mini_wrapper(tmp_path, **srcs))
    assert any(message in e for e in errors), errors
    assert verify_profiler_contract.main([mini_wrapper(tmp_path / "again", **srcs)]) == 1


def test_contract_close_before_decode_is_rejected(tmp_path):
    loop = MINI_LOOP.replace("        sampling_pbar.close()\n", "").replace(
        "        vae.to(device)\n        videos", "        sampling_pbar.close()\n        vae.to(device)\n        videos")
    errors = verify_profiler_contract.check(mini_wrapper(tmp_path, loop=loop))
    assert any("no longer closes after vae.decode" in e for e in errors), errors


@pytest.mark.skipif(not os.environ.get("WRAPPER_SRC"), reason="set WRAPPER_SRC to an installed wrapper checkout")
def test_contract_on_real_wrapper_source():
    assert verify_profiler_contract.check(os.environ["WRAPPER_SRC"]) == []
