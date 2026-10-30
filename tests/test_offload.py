# ==============================================================================
# Offloading OFF by default (worker_guard/offload.py + job_guard integration):
# blocks_to_swap 0 / force_offload False at any resolution, client values ignored,
# MIN_VRAM_GB (28) rejection per job, manual mode only with OFFLOAD_ALLOW_MANUAL=1,
# the OOM ladder (+10, 25, 40, then 40 with VAE tiling) and its time check, OFFLOAD_TIERS
# override, peak VRAM, and the near-zero watchdog rule when blocks are swapped.
# ComfyUI is fully mocked.
# ==============================================================================
import json
import logging

import pytest

from worker_guard import comfy_api, job_guard, offload

GB = 1024 ** 3
MB = 1024 ** 2
OOM_DETAILS = {
    "type": "execution_error", "node_id": "128", "node_type": "WanVideoSampler",
    "exception_type": "torch.OutOfMemoryError",
    "exception_message": "Allocation on device", "traceback": [],
}
RTX5090 = "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync"


def oom(node_type="WanVideoSampler", exception_type="torch.OutOfMemoryError"):
    details = {**OOM_DETAILS, "node_type": node_type, "exception_type": exception_type}
    return {"error": f"ComfyUI execution_error at node 128 ({node_type}): {exception_type}: Allocation on device",
            "details": details}


def device(total_gb, used_mb=900, name=RTX5090, dtype="cuda"):
    total = int(total_gb * GB)
    return {"devices": [{"type": dtype, "name": name, "index": 0, "vram_total": total,
                         "vram_free": total - used_mb * MB}]}


class FakeClock:
    def __init__(self):
        self.t = 5000.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(job_guard, "_now", c.now)
    monkeypatch.setattr(job_guard, "_sleep", c.sleep)
    return c


@pytest.fixture
def gpu(monkeypatch):
    """Fake ComfyUI: set .stats per test; records /free calls and the order of events."""
    class State:
        stats = device(31.4)
        after_free = None

    state = State()
    state.events = []

    def system_stats(timeout=10):
        return state.stats

    def free(unload_models=True, free_memory=True, timeout=10):
        state.events.append(("free", unload_models, free_memory))
        if state.after_free is not None:
            state.stats = state.after_free

    monkeypatch.setattr(comfy_api, "system_stats", system_stats)
    monkeypatch.setattr(comfy_api, "free", free)
    monkeypatch.setattr(comfy_api, "interrupt", lambda prompt_id=None, timeout=10: state.events.append(("interrupt",)))
    return state


def recording_handler(results, state):
    """Inner handler that records the job input it receives and returns scripted results."""
    def inner(job):
        inp = job["input"]
        state.events.append(("run", inp.get("blocks_to_swap"), inp.get("force_offload"), inp.get("load_device")))
        return results.pop(0)
    return inner


def guarded(results, state):
    gh, _ = job_guard.install(recording_handler(results, state), lambda *a, **k: {})
    return gh


def runs(state):
    return [e[1:] for e in state.events if e[0] == "run"]


# ---------------------------------------------------------------- decision: offloading off (except 720p < 44 GB)
@pytest.mark.parametrize("vram_gb", [28.0, 31.4, 43.99, 47.5, 79.2])
@pytest.mark.parametrize("size", [(512, 512), (720, 480), (832, 480), (900, 500)])
def test_480p_offloading_off_on_every_gpu(vram_gb, size):
    d, err = offload.decide({"width": size[0], "height": size[1]}, vram_gb * GB)
    assert err is None
    assert (d.mode, d.row, d.blocks_to_swap, d.force_offload) == ("auto", "480p", 0, False)


@pytest.mark.parametrize("vram_gb, expected", [
    (28.0, (25, True)), (31.4, (25, True)), (43.99, (25, True)),   # 32 GB RTX 5090 ran out of memory at 720p
    (44.0, (0, False)), (47.5, (0, False)), (79.2, (0, False)),
])
@pytest.mark.parametrize("size", [(901, 500), (1280, 720), (1920, 1080)])
def test_720p_swaps_below_44gb(vram_gb, expected, size):
    d, err = offload.decide({"width": size[0], "height": size[1]}, vram_gb * GB)
    assert err is None and d.row == "720p"
    assert (d.blocks_to_swap, d.force_offload) == expected


@pytest.mark.parametrize("vram_gb, ok", [(28.0, True), (27.99, False), (23.6, False), (19.6, False), (8, False)])
def test_min_vram_28(vram_gb, ok):
    d, err = offload.decide({"width": 720, "height": 480}, vram_gb * GB, "NVIDIA GeForce RTX 4090")
    assert (err is None) is ok
    if not ok:
        assert d is None
        assert err == f"[OFFLOAD] NVIDIA GeForce RTX 4090 has {vram_gb:.1f} GB VRAM; minimum is 28 GB. Job not run."


def test_min_vram_env_override(monkeypatch):
    monkeypatch.setenv("MIN_VRAM_GB", "40")
    assert offload.decide({}, 31.4 * GB)[1].endswith("minimum is 40 GB. Job not run.")
    assert offload.decide({}, 47.5 * GB)[1] is None
    monkeypatch.setenv("MIN_VRAM_GB", "abc")  # invalid -> default 28
    assert offload.decide({}, 31.4 * GB)[1] is None


def test_min_vram_wins_over_lower_custom_tier(monkeypatch):
    monkeypatch.setenv("OFFLOAD_TIERS", json.dumps([
        {"min_vram_gb": 20, "480p": {"blocks_to_swap": 20, "force_offload": True},
         "720p": {"blocks_to_swap": 40, "force_offload": True}}]))
    assert offload.decide({}, 23.6 * GB)[1].endswith("minimum is 28 GB. Job not run.")


def test_client_values_ignored():
    job_input = {"blocks_to_swap": 40, "force_offload": True, "load_device": "offload_device",
                 "width": 1280, "height": 720}
    d, _ = offload.decide(job_input, 31.4 * GB)
    out = offload.apply(job_input, d)
    assert out["blocks_to_swap"] == 25 and out["force_offload"] is True and "load_device" not in out
    assert job_input["blocks_to_swap"] == 40  # client input not mutated
    small = {"blocks_to_swap": 40, "force_offload": True, "width": 720, "height": 480}
    out = offload.apply(small, offload.decide(small, 31.4 * GB)[0])
    assert out["blocks_to_swap"] == 0 and out["force_offload"] is False


def test_manual_mode_ignored_by_default(caplog):
    job_input = {"offload_mode": "manual", "blocks_to_swap": 20, "force_offload": True}
    with caplog.at_level(logging.WARNING, logger="worker_guard"):
        d, err = offload.decide(job_input, 31.4 * GB)
    assert err is None and (d.mode, d.blocks_to_swap, d.force_offload) == ("auto", 0, False)
    assert "[OFFLOAD] manual mode ignored (OFFLOAD_ALLOW_MANUAL=0); client offload values are not used" in caplog.messages
    assert offload.apply(job_input, d)["blocks_to_swap"] == 0


def test_manual_mode_with_allow_env(monkeypatch):
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
    job_input = {"offload_mode": "Manual", "blocks_to_swap": "7", "force_offload": False, "load_device": "main_device",
                 "width": 1280, "height": 720}
    d, err = offload.decide(job_input, 31.4 * GB)
    assert err is None and (d.mode, d.blocks_to_swap, d.force_offload) == ("manual", 7, False)
    assert offload.apply(job_input, d) is job_input
    assert offload.decide(job_input, 23.6 * GB)[0] is None  # minimum still applies in manual mode


@pytest.mark.parametrize("vram_gb", [28.0, 31.4, 47.5, 79.2])
def test_manual_mode_never_offloads_480p_on_28gb_plus(monkeypatch, caplog, vram_gb):
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
    job_input = {"offload_mode": "manual", "blocks_to_swap": 38, "force_offload": True, "load_device": "offload_device",
                 "width": 854, "height": 480}
    with caplog.at_level(logging.WARNING, logger="worker_guard"):
        d, err = offload.decide(job_input, vram_gb * GB)
    assert err is None and (d.mode, d.blocks_to_swap, d.force_offload) == ("auto", 0, False)
    assert f"[OFFLOAD] manual values ignored: 480p on {vram_gb:.1f} GB always runs with blocks_to_swap 0, " \
           "force_offload False" in caplog.messages
    out = offload.apply(job_input, d)
    assert (out["blocks_to_swap"], out["force_offload"], "load_device" in out) == (0, False, False)


def test_offload_tiers_cannot_offload_480p_on_28gb_plus(monkeypatch, caplog):
    monkeypatch.setenv("OFFLOAD_TIERS", json.dumps([
        {"min_vram_gb": 28, "480p": {"blocks_to_swap": 38, "force_offload": True},
         "720p": {"blocks_to_swap": 30, "force_offload": True}}]))
    with caplog.at_level(logging.WARNING, logger="worker_guard"):
        d, _ = offload.decide({"width": 854, "height": 480}, 31.4 * GB)
    assert (d.blocks_to_swap, d.force_offload) == (0, False)
    assert any("OFFLOAD_TIERS 480p row" in m and "ignored" in m for m in caplog.messages)
    d, _ = offload.decide({"width": 1280, "height": 720}, 31.4 * GB)  # 720p still follows the tiers
    assert (d.blocks_to_swap, d.force_offload) == (30, True)


def test_unknown_mode_falls_back_to_auto():
    assert offload.mode_of({"offload_mode": "whatever"}) == "auto"
    assert offload.mode_of({}) == "auto"


def test_missing_size_uses_handler_default_512():
    d, _ = offload.decide({}, 31.4 * GB)
    assert (d.width, d.height, d.row) == (512, 512, "480p")


# ---------------------------------------------------------------- OFFLOAD_TIERS / env config
def test_offload_tiers_env_override(monkeypatch):
    monkeypatch.setenv("OFFLOAD_TIERS", json.dumps([
        {"min_vram_gb": 28, "480p": {"blocks_to_swap": 0, "force_offload": False},
         "720p": {"blocks_to_swap": 6, "force_offload": False}},
        {"min_vram_gb": 44, "480p": {"blocks_to_swap": 0, "force_offload": False},
         "720p": {"blocks_to_swap": 0, "force_offload": False}},
    ]))
    d, _ = offload.decide({"width": 1280, "height": 720}, 31.4 * GB)
    assert (d.blocks_to_swap, d.force_offload) == (6, False)
    d, _ = offload.decide({"width": 1280, "height": 720}, 47.5 * GB)  # unsorted input still works
    assert (d.blocks_to_swap, d.force_offload) == (0, False)


def test_custom_tier_above_minimum_still_rejects_gaps(monkeypatch):
    monkeypatch.setenv("OFFLOAD_TIERS", json.dumps([
        {"min_vram_gb": 40, "480p": {"blocks_to_swap": 0, "force_offload": False},
         "720p": {"blocks_to_swap": 0, "force_offload": False}}]))
    d, err = offload.decide({}, 31.4 * GB)
    assert d is None and "needs at least 40 GB" in err


@pytest.mark.parametrize("raw", [
    "not json",
    "[]",
    '{"min_vram_gb": 28}',
    '[{"min_vram_gb": 28, "480p": {"blocks_to_swap": 41, "force_offload": false}, "720p": {"blocks_to_swap": 0, "force_offload": false}}]',
    '[{"min_vram_gb": 28, "480p": {"blocks_to_swap": 0, "force_offload": "no"}, "720p": {"blocks_to_swap": 0, "force_offload": false}}]',
    '[{"min_vram_gb": 28, "480p": {"blocks_to_swap": 0, "force_offload": false}}]',
    '[{"min_vram_gb": true, "480p": {"blocks_to_swap": 0, "force_offload": false}, "720p": {"blocks_to_swap": 0, "force_offload": false}}]',
])
def test_invalid_offload_tiers_fall_back_to_defaults(monkeypatch, caplog, raw):
    monkeypatch.setenv("OFFLOAD_TIERS", raw)
    with caplog.at_level(logging.ERROR, logger="worker_guard"):
        assert offload.load_tiers() == offload.validate_tiers(offload.DEFAULT_TIERS)
    assert any("invalid OFFLOAD_TIERS" in r.getMessage() for r in caplog.records)


def test_default_tiers():
    assert offload.validate_tiers(offload.DEFAULT_TIERS) == [
        {"min_vram_gb": 44.0, "480p": {"blocks_to_swap": 0, "force_offload": False},
         "720p": {"blocks_to_swap": 0, "force_offload": False}},
        {"min_vram_gb": 28.0, "480p": {"blocks_to_swap": 0, "force_offload": False},
         "720p": {"blocks_to_swap": 25, "force_offload": True}}]


def test_480p_pixel_boundary(monkeypatch):
    assert offload.resolution_row(900, 500) == "480p"
    assert offload.resolution_row(901, 500) == "720p"
    monkeypatch.setenv("OFFLOAD_480P_MAX_PIXELS", str(1280 * 720))
    assert offload.resolution_row(1280, 720) == "480p"


# ---------------------------------------------------------------- OOM detection / emergency settings
def test_is_oom():
    assert offload.is_oom(oom())
    assert offload.is_oom(oom(node_type="WanVideoDecode"))
    assert offload.is_oom(oom(exception_type="torch.cuda.OutOfMemoryError"))
    assert not offload.is_oom(oom(node_type="WanVideoTextEncodeCached"))
    assert not offload.is_oom(oom(exception_type="RuntimeError"))
    assert not offload.is_oom({"error": "torch.OutOfMemoryError somewhere"})
    assert not offload.is_oom({"video": "b64"})
    assert not offload.is_oom(None)


@pytest.mark.parametrize("current, expected", [
    (0, [(10, False), (25, False), (40, False), (40, True)]),
    (6, [(16, False), (25, False), (40, False), (40, True)]),
    (20, [(30, False), (40, False), (40, True)]),    # 25 is not more than 30: skipped
    (25, [(35, False), (40, False), (40, True)]),
    (35, [(40, False), (40, True)]),
    (40, [(40, True)]),                               # already at 40: only the VAE tiling step is left
])
def test_oom_ladder(current, expected):
    steps = offload.oom_ladder(current)
    assert steps == expected
    # VAE tiling is always the last step and always comes right after (or at) 40 blocks
    assert [t for _, t in steps] == [False] * (len(steps) - 1) + [True]
    assert steps[-1] == (40, True) and (len(steps) == 1 or steps[-2] == (40, False))


# ---------------------------------------------------------------- integration through the guard
def test_job_runs_with_offloading_off_and_logs(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([{"video": "b64"}], gpu)
    res = gh({"input": {"width": 720, "height": 480, "blocks_to_swap": 25, "force_offload": True,
                        "load_device": "offload_device"}})
    assert res == {"video": "b64"}
    assert runs(gpu) == [(0, False, None)]
    assert "[OFFLOAD] auto: VRAM 31.4 GB, 720x480 -> blocks_to_swap 0, force_offload False" in caplog.messages


def test_720p_job_on_32gb_swaps_25_blocks(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([{"video": "b64"}], gpu)
    assert gh({"input": {"width": 1280, "height": 720, "blocks_to_swap": 0, "force_offload": False}}) == {"video": "b64"}
    assert runs(gpu) == [(25, True, None)]
    assert "[OFFLOAD] auto: VRAM 31.4 GB, 1280x720 -> blocks_to_swap 25, force_offload True" in caplog.messages


def test_720p_job_on_48gb_has_no_offloading(gpu, clock):
    gpu.stats = device(47.5, name="cuda:0 NVIDIA L40S : cudaMallocAsync")
    guarded([{"video": "b64"}], gpu)({"input": {"width": 1280, "height": 720}})
    assert runs(gpu) == [(0, False, None)]


@pytest.mark.parametrize("job_input", [
    {"width": 512, "height": 512, "person_count": "multi"},
    {"offload_mode": "manual", "blocks_to_swap": 40},
    {"task": "tts", "text": "hello"},
])
def test_below_28gb_rejected_for_every_job(gpu, clock, job_input):
    gpu.stats = device(23.6, name="cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync")
    gh = guarded([{"video": "b64"}], gpu)
    res = gh({"input": job_input})
    assert res == {"error": "[OFFLOAD] NVIDIA GeForce RTX 4090 has 23.6 GB VRAM; minimum is 28 GB. Job not run.",
                   "refresh_worker": True}
    assert runs(gpu) == []  # nothing ran (and the Multi model was not copied)


def test_below_28gb_rejected_even_with_manual_allowed(gpu, clock, monkeypatch):
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
    gpu.stats = device(23.6)
    res = guarded([{"video": "b64"}], gpu)({"input": {"offload_mode": "manual", "blocks_to_swap": 40}})
    assert res["refresh_worker"] is True and runs(gpu) == []


def test_cmd_diagnostics_still_run_on_small_gpu(gpu, clock):
    gpu.stats = device(8)
    assert guarded([{"cmd": "x", "output": ""}], gpu)({"input": {"cmd": "nvidia-smi"}}) == {"cmd": "x", "output": ""}


def test_manual_job_passes_client_values_when_allowed(gpu, clock, monkeypatch, caplog):
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([{"video": "b64"}], gpu)
    gh({"input": {"offload_mode": "manual", "blocks_to_swap": 40, "force_offload": True, "load_device": "offload_device",
                  "width": 1280, "height": 720}})
    assert runs(gpu) == [(40, True, "offload_device")]
    assert "[OFFLOAD] manual: blocks_to_swap 40, force_offload True (as sent)" in caplog.messages


def test_ladder_first_step_succeeds(gpu, clock, caplog):
    gpu.stats = device(31.4, used_mb=30000)
    gpu.after_free = device(31.4, used_mb=700)
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([oom(), {"video": "b64"}], gpu)
    res = gh({"input": {"width": 720, "height": 480}})
    assert res == {"video": "b64"}  # no emergency_vae_tiling key: tiling never ran
    # first run without offloading, then /free, then the first ladder step: 10 swapped blocks
    assert gpu.events == [("run", 0, False, None), ("free", True, True), ("run", 10, False, None)]
    assert any(m.startswith("[OFFLOAD] OOM retry 1/4: out of memory at sampler (transformer steps, node 128 "
                            "WanVideoSampler, window n/a); blocks_to_swap 0 -> 10, force_offload False")
               for m in caplog.messages)
    assert "[OFFLOAD] /free done: VRAM used 0.7 GB" in caplog.messages
    assert not any("VAE tiling" in m for m in caplog.messages)
    job_line = [m for m in caplog.messages if m.startswith("[JOB] done in ")][0]
    assert "blocks_to_swap 10, force_offload False, mode auto, oom_retry yes" in job_line


def test_ladder_at_720p_on_32gb_goes_25_35(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([oom(), {"video": "b64"}], gpu)
    assert gh({"input": {"width": 1280, "height": 720}}) == {"video": "b64"}
    assert runs(gpu) == [(25, True, None), (35, True, None)]
    assert any(m.startswith("[OFFLOAD] OOM retry 1/3: out of memory at sampler") and "blocks_to_swap 25 -> 35, "
               "force_offload True" in m for m in caplog.messages)


def test_ladder_also_for_vae_decode_oom(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    gh = guarded([oom(node_type="WanVideoDecode"), {"video": "b64"}], gpu)
    assert gh({"input": {}}) == {"video": "b64"}
    assert runs(gpu) == [(0, False, None), (10, False, None)]
    assert any("out of memory at vae_decode (node 128 WanVideoDecode)" in m for m in caplog.messages)


def test_ladder_walks_every_step_then_vae_tiling(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    seen_tiling = []

    def inner(job):
        inp = job["input"]
        seen_tiling.append(inp.get("vae_tiling", False))
        gpu.events.append(("run", inp.get("blocks_to_swap"), inp.get("force_offload"), inp.get("load_device")))
        return results.pop(0)

    results = [oom(), oom(), oom(), oom(), {"video": "b64"}]
    gh, _ = job_guard.install(inner, lambda *a, **k: {})
    res = gh({"input": {"width": 720, "height": 480}})
    assert runs(gpu) == [(0, False, None), (10, False, None), (25, False, None), (40, False, None), (40, False, None)]
    # VAE tiling only on the very last attempt, after the 40-block attempt also ran out of memory
    assert seen_tiling == [False, False, False, False, True]
    assert res == {"video": "b64", "emergency_vae_tiling": True}
    assert "[OFFLOAD] OOM retry final: VAE tiling ON (output may differ)" in caplog.messages
    final = caplog.messages.index("[OFFLOAD] OOM retry final: VAE tiling ON (output may differ)")
    assert any("[OFFLOAD] OOM retry 3/4" in m for m in caplog.messages[:final])


def test_ladder_failure_after_every_step(gpu, clock):
    gh = guarded([oom(), oom(), oom(), oom()], gpu)
    res = gh({"input": {"width": 1280, "height": 720}})
    assert runs(gpu) == [(25, True, None), (35, True, None), (40, True, None), (40, True, None)]
    assert res["refresh_worker"] is True
    assert res["emergency_vae_tiling"] is True
    assert res["error"].startswith(
        "[OFFLOAD] out of memory again after OOM retry: GPU NVIDIA GeForce RTX 5090, VRAM 31.4 GB, "
        "1280x720, frames n/a, blocks_to_swap 40, force_offload True; ladder 25 -> 35 -> 40 -> 40 (VAE tiling)"
    )
    assert res["offload"] == {"gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 31.4, "width": 1280, "height": 720,
                              "frames": None, "blocks_to_swap": 40, "force_offload": True, "mode": "auto",
                              "oom_retry": True}
    assert res["details"]["exception_type"] == "torch.OutOfMemoryError"


def test_retry_failing_with_other_error_stops_the_ladder(gpu, clock):
    res = guarded([oom(), {"error": "FileNotFoundError: x"}, {"video": "never"}], gpu)({"input": {}})
    assert len(runs(gpu)) == 2
    assert res["error"].startswith("[OFFLOAD] failed again after OOM retry")
    assert res["error"].endswith("| ComfyUI: FileNotFoundError: x")
    assert res["refresh_worker"] is True and "emergency_vae_tiling" not in res


def test_ladder_stops_at_success(gpu, clock):
    guarded([oom(), oom(), {"video": "b64"}, {"video": "never"}], gpu)({"input": {}})
    assert runs(gpu) == [(0, False, None), (10, False, None), (25, False, None)]


@pytest.mark.parametrize("env, expected_limit", [
    ({"JOB_TIMEOUT_SEC": "100"}, "JOB_TIMEOUT_SEC=100s"),
    ({"RUNPOD_EXECUTION_TIMEOUT_SEC": "200"}, "RUNPOD_EXECUTION_TIMEOUT_SEC=200s minus 120s to return the video"),
])
def test_ladder_not_started_without_time(gpu, clock, caplog, monkeypatch, env, expected_limit):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    caplog.set_level(logging.INFO, logger="worker_guard")
    res = guarded([oom(), {"video": "never"}], gpu)({"input": {}})
    assert len(runs(gpu)) == 1  # the retry needs OOM_RETRY_MIN_SEC (120s) at least; under 120s are left
    assert f"before {expected_limit}" in res["error"]
    assert "OOM retry 1/4 not started: needs ~120s" in res["error"]
    assert res["refresh_worker"] is True


def test_ladder_time_check_runs_before_every_step(gpu, clock, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SEC", "400")

    def inner(job):
        gpu.events.append(("run", job["input"].get("blocks_to_swap"), None, None))
        clock.t += 100  # every attempt takes 100 s: a retry is estimated at 2 x 100 + 30 (free wait)
        return oom()

    gh, _ = job_guard.install(inner, lambda *a, **k: {})
    res = gh({"input": {}})
    # 0 s: run 1 (ends at 100); 300 s left >= 230 -> run 2 (ends at 200); 200 s left < 230 -> stop
    assert [e[1] for e in gpu.events if e[0] == "run"] == [0, 10]
    assert "OOM retry 2/4 not started: needs ~230s" in res["error"]


def test_no_retry_for_manual_mode(gpu, clock, monkeypatch):
    monkeypatch.setenv("OFFLOAD_ALLOW_MANUAL", "1")
    res = guarded([oom()], gpu)({"input": {"offload_mode": "manual", "blocks_to_swap": 0, "width": 1280, "height": 720}})
    assert runs(gpu) == [(0, None, None)]
    assert res == oom() and "refresh_worker" not in res


def test_no_retry_for_oom_in_other_nodes(gpu, clock):
    res = guarded([oom(node_type="WanVideoTextEncodeCached")], gpu)({"input": {}})
    assert len(runs(gpu)) == 1 and res["details"]["node_type"] == "WanVideoTextEncodeCached"


def test_free_wait_gives_up_after_timeout(gpu, clock, caplog):
    gpu.stats = device(31.4, used_mb=20000)  # /free never takes effect
    caplog.set_level(logging.INFO, logger="worker_guard")
    start = clock.t
    assert guarded([oom(), {"video": "b64"}], gpu)({"input": {}}) == {"video": "b64"}
    assert clock.t - start >= 30
    assert any(m.startswith("[OFFLOAD] VRAM used still 19.5 GB 30s after /free; retrying anyway") for m in caplog.messages)


# ---------------------------------------------------------------- watchdog with swapped blocks
PROMPT = {
    "122": {"class_type": "WanVideoModelLoader", "inputs": {}},
    "128": {"class_type": "WanVideoSampler", "inputs": {}},
    "270": {"class_type": "INTConstant", "inputs": {"value": 181}},
}


def watchdog_ctx(blocks):
    ctx = job_guard.JobContext(job_guard.Settings.from_env(), "video")
    ctx.record_prompt(PROMPT)
    ctx.offload = offload.Decision("auto", blocks, False, 31.4, 512, 512, "480p")
    ctx.executing = True
    return ctx


def run_for(wd, clock, seconds, step=5):
    end = clock.t + seconds
    while clock.t < end:
        clock.t += step
        wd.tick()


@pytest.mark.parametrize("blocks, node, expect_trip", [
    (6, "122", False),   # emergency swap: loader fills CPU RAM, exempt
    (0, "122", True),    # offloading off (default): loader loads to the GPU, rule applies
    (6, "128", True),    # sampler is never exempt
])
def test_near_zero_rule_with_swapped_blocks(gpu, clock, blocks, node, expect_trip):
    gpu.stats = device(31.4, used_mb=400)
    ctx = watchdog_ctx(blocks)
    ctx.current_node = node
    wd = job_guard.Watchdog(ctx, None)
    run_for(wd, clock, 240)
    assert (ctx.trip_kind == "vram_idle") is expect_trip


def test_peak_vram_sampled_every_5s_logged_every_30s(gpu, clock, caplog):
    caplog.set_level(logging.INFO, logger="worker_guard")
    ctx = watchdog_ctx(0)
    ctx.current_node = "128"
    wd = job_guard.Watchdog(ctx, None)
    for used in (3000, 9000, 22000, 7000, 4000, 4000):
        gpu.stats = device(31.4, used_mb=used)
        clock.t += 5
        wd.tick()
    assert ctx.peak_vram == 22000 * MB
    assert len([m for m in caplog.messages if m.startswith("[WATCHDOG] VRAM used")]) == 1
