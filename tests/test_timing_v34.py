"""v34: [TIMING] stages + the timing dict benchmark jobs return."""
from types import SimpleNamespace

from worker_guard.node_timer import NodeTimer
from worker_guard.speed import SpeedTracker
from worker_guard import job_guard

CLASSES = {"302": "MelBandRoFormerSampler", "194": "MultiTalkWav2VecEmbeds", "241": "WanVideoTextEncodeCached",
           "237": "WanVideoClipVisionEncode", "122": "WanVideoModelLoader", "192": "WanVideoImageToVideoMultiTalk",
           "128": "WanVideoSampler", "130": "WanVideoDecode", "131": "VHS_VideoCombine", "999": "PreviewAny"}


def run_nodes(timer, plan):
    t = 0.0
    for node, sec in plan:
        timer.on_executing(node, t)
        t += sec
    timer.on_executing(None, t)


def test_stages_group_nodes_by_class(caplog):
    timer = NodeTimer()
    run_nodes(timer, [("302", 4), ("194", 3), ("241", 6), ("237", 1), ("122", 20), ("192", 2), ("128", 300),
                      ("130", 5), ("131", 2), ("999", 0.5)])
    stages = timer.stages(CLASSES)
    assert stages == {"vocal_separation": 4, "wav2vec": 3, "t5": 6, "clip": 1, "model_load": 20, "image_embeds": 2,
                      "sampler": 300, "vae_decode": 5, "video_encode": 2, "other": 0.5}
    assert list(stages) == ["vocal_separation", "wav2vec", "t5", "clip", "model_load", "image_embeds", "sampler",
                            "vae_decode", "video_encode", "other"]
    caplog.set_level("INFO", logger="worker_guard")
    timer.log_stages(CLASSES)
    assert "[TIMING] stages: vocal_separation 4.0s" in caplog.text


def test_job_timing_has_windows_and_stages():
    sp = SpeedTracker(6, "128")
    sp.on_executing("128", 0.0)
    t = 0.0
    for step in range(1, 25):              # 4 windows x 6 steps, 10 s per step
        t += 10.0
        sp.on_progress("128", step, 24, t)
    timer = NodeTimer()
    run_nodes(timer, [("194", 3), ("128", 240), ("131", 2)])
    ctx = SimpleNamespace(speed=sp, nodes=timer, node_classes=CLASSES, gen_sec=245.0, fp8_fast=True)
    out = job_guard.job_timing(ctx)
    assert [w["index"] for w in out["windows"]] == [1, 2, 3, 4]
    assert all(w["sec"] == 60.0 and w["s_per_it"] == 10.0 for w in out["windows"])
    assert out["steps_per_window"] == 6 and out["avg_s_per_it"] == 10.0
    assert out["stages_sec"] == {"wav2vec": 3, "sampler": 240, "video_encode": 2}
    assert out["comfy_total_sec"] == 245.0 and out["fp8_fast"] is True


def test_job_timing_without_trackers():
    out = job_guard.job_timing(SimpleNamespace(node_classes={}))
    assert out["windows"] == [] and out["stages_sec"] == {} and out["avg_s_per_it"] is None
