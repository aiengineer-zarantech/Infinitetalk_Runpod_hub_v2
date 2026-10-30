# ==============================================================================
# comfy_vram/stages.py (ComfyUI side, no ComfyUI or GPU here): stage tagging, peaks per stage,
# VAE wrappers (result unchanged, OOM recorded with node and window), VAE class lookup, on_prompt.
# ==============================================================================
import sys
import types

import pytest

from comfy_vram import stages

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(stages, "STATE", stages.StageState())
    monkeypatch.setattr(stages, "_vae_kind", None)
    monkeypatch.setattr(stages, "_vae_patched", None)
    node = {"id": None}
    monkeypatch.setattr(stages, "_current_node", lambda: node["id"])
    return node


CLASSES = {"128": "WanVideoSampler", "130": "WanVideoDecode", "241": "WanVideoTextEncodeCached"}


def test_stage_for():
    s = stages.StageState()
    s.reset(CLASSES, 32 * GB, GB)
    assert s.stage_for("128", None) == "sampler"
    assert s.stage_for("128", "vae_decode") == "vae_decode"
    assert s.stage_for("130", None) == "vae_decode"
    assert s.stage_for("241", None) == "other"
    assert s.stage_for(None, None) is None


def test_peaks_per_stage_and_node():
    s = stages.StageState()
    s.reset(CLASSES, 32 * GB, GB)
    for used, node, kind in [(10, "128", None), (20, "128", None), (15, "128", "vae_decode"),
                             (25, "128", "vae_decode"), (5, "241", None), (30, None, None)]:
        s.sample(used * GB, 32 * GB, node, kind)
    snap = s.snapshot()
    assert snap["stages"] == {"sampler": 20 * GB, "vae_decode": 25 * GB, "other": 5 * GB}
    assert snap["nodes"] == {"128": 25 * GB, "241": 5 * GB}
    assert snap["samples"] == 5 and snap["total"] == 32 * GB and snap["baseline"] == GB
    s.reset(CLASSES, 32 * GB, 2 * GB)
    assert s.snapshot()["stages"] == {} and s.snapshot()["samples"] == 0


class FakeVAE:
    def __init__(self):
        self.calls = []

    def encode(self, videos, device, tiled=False):
        self.calls.append(("encode", stages._vae_kind))
        return ("latent", videos)

    def decode(self, hidden_states, device, tiled=False):
        self.calls.append(("decode", stages._vae_kind))
        if hidden_states == "boom":
            raise FakeOOM("CUDA out of memory")
        return ("frames", hidden_states)


class FakeOOM(RuntimeError):
    pass


FakeOOM.__name__ = "OutOfMemoryError"


def patched_vae_module(monkeypatch):
    module = types.ModuleType("fake_vae")
    module.__file__ = "/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/wanvideo/wan_video_vae.py"

    class WanVideoVAE(FakeVAE):
        pass

    module.WanVideoVAE = WanVideoVAE
    assert stages.patch_vae({"fake_vae": module}) is True
    return WanVideoVAE


def test_vae_wrappers_return_the_same_result_and_tag_the_stage(monkeypatch, fresh):
    cls = patched_vae_module(monkeypatch)
    stages.STATE.reset(CLASSES, 32 * GB, GB)
    fresh["id"] = "128"
    vae = cls()
    assert vae.encode("v", "cuda", tiled=True) == ("latent", "v")
    assert vae.decode("z", "cuda") == ("frames", "z")
    assert vae.calls == [("encode", "vae_encode"), ("decode", "vae_decode")]
    assert stages._vae_kind is None  # reset after the call
    assert stages.STATE.snapshot()["decodes"] == {"128": 1}
    assert stages.STATE.snapshot()["vae_hooks"] is True


def test_oom_in_vae_decode_is_recorded_with_window(monkeypatch, fresh):
    cls = patched_vae_module(monkeypatch)
    stages.STATE.reset(CLASSES, 32 * GB, GB)
    fresh["id"] = "128"
    vae = cls()
    vae.decode("z1", "cuda")
    vae.decode("z2", "cuda")
    with pytest.raises(FakeOOM):
        vae.decode("boom", "cuda")  # re-raised unchanged
    assert stages.STATE.snapshot()["oom"] == {"stage": "vae_decode", "node_id": "128",
                                              "node_type": "WanVideoSampler", "window": 3}
    assert stages._vae_kind is None


def test_patch_vae_without_the_wrapper_module(caplog):
    assert stages.patch_vae({}) is False
    assert any("VAE stages off" in m for m in caplog.messages)
    assert stages.STATE.snapshot()["vae_hooks"] is False


def test_on_prompt_resets_and_starts_the_sampler_once(monkeypatch):
    started = []

    class FakeSampler:
        def __init__(self, mem_get_info, interval):
            self.mem_get_info = mem_get_info

        def start(self):
            started.append(1)

    monkeypatch.setattr(stages, "Sampler", FakeSampler)
    monkeypatch.setattr(stages, "_sampler", None)
    monkeypatch.setattr(stages, "_mem_get_info", lambda: (20 * GB, 32 * GB))
    monkeypatch.setattr(stages, "patch_vae", lambda modules=None: False)
    stages.STATE.sample(30 * GB, 32 * GB, "128", None)
    data = {"prompt": {"128": {"class_type": "WanVideoSampler", "inputs": {}}}}
    assert stages.on_prompt(data) is data
    assert stages.on_prompt(data) is data
    snap = stages.STATE.snapshot()
    assert snap["stages"] == {} and snap["baseline"] == 12 * GB and snap["total"] == 32 * GB
    assert started == [1]


def test_sampler_thread_sample_uses_current_node_and_vae_kind(monkeypatch, fresh):
    stages.STATE.reset(CLASSES, 32 * GB, GB)
    sampler = stages.Sampler(lambda: (8 * GB, 32 * GB), 0.05)
    sampler._sample("128", None)
    sampler._sample("128", "vae_decode")
    assert stages.STATE.snapshot()["stages"] == {"sampler": 24 * GB, "vae_decode": 24 * GB}


def test_package_off_with_env(monkeypatch):
    monkeypatch.setenv("VRAM_STAGES", "0")
    sys.modules.pop("comfy_vram", None)
    import importlib
    pkg = importlib.import_module("comfy_vram")
    assert pkg.NODE_CLASS_MAPPINGS == {} and not pkg._enabled()
