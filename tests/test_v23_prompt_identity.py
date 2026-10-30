# ==============================================================================
# Old jobs are unchanged by v24: the same jobs run through the REAL handler.py + worker_guard of
# tag v1.0-stable (= image v23, taken from git) and of the current tree, ComfyUI mocked, must POST
# exactly the same workflows to /prompt and return exactly the same results.
# ==============================================================================
import base64
import io
import json
import shutil
import subprocess
import sys
import tarfile
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "tests" / "fixtures" / "prompt_driver.py"
TAG = "v1.0-stable"
TREE_FILES = ["handler.py", "setup_models.py", "worker_guard", "I2V_single.json", "I2V_multi.json",
              "V2V_single.json", "V2V_multi.json", "I2V_single_compiled.json", "I2V_multi_compiled.json"]


def _wav_b64(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


IMAGE = base64.b64encode(b"\x89PNG fake image").decode()
WAV = base64.b64encode(b"RIFF fake wav").decode()
# What the studio backend sends today (backend/main.py), plus older and test-only inputs
STUDIO = {"input_type": "image", "person_count": "single", "prompt": "A person talking naturally",
          "width": 720, "height": 480, "blocks_to_swap": 0, "force_offload": False, "network_volume": False,
          "image_base64": IMAGE, "wav_base64": WAV}
JOBS = [
    ({}, STUDIO),
    ({}, {**STUDIO, "width": 1280, "height": 720}),
    ({}, {**STUDIO, "width": 854, "height": 480}),
    ({}, {"image_base64": IMAGE, "wav_base64": WAV}),  # no size: handler default 512x512
    ({}, {**STUDIO, "input_type": "video", "video_base64": base64.b64encode(b"fake mp4").decode()}),
    ({}, {**STUDIO, "person_count": "multi"}),
    ({}, {**STUDIO, "max_frame": 121}),
    ({}, {**STUDIO, "width": 1280, "height": 720, "vae_tiling": True}),
    ({"TORCH_COMPILE": "0"}, STUDIO),
    ({"FP8_FAST": "1"}, STUDIO),
    ({}, {**STUDIO, "benchmark": True, "torch_compile": False, "wav_base64": _wav_b64(12)}),
    ({"OFFLOAD_ALLOW_MANUAL": "1"}, {**STUDIO, "offload_mode": "manual", "blocks_to_swap": 10,
                                     "force_offload": True, "width": 1280, "height": 720}),
    ({}, STUDIO),  # same job again on the same worker
]


def _git_tree(tag, dest):
    try:
        data = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", tag, *TREE_FILES],
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git archive {tag} not available: {e}")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)


def _run(root, tmp_path, name, job_list=JOBS):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(job_list), encoding="utf-8")
    out = tmp_path / f"{name}.json"
    proc = subprocess.run([sys.executable, str(DRIVER), str(root), str(jobs), str(out)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_old_jobs_post_identical_workflows_and_results(tmp_path):
    old_root, new_root = tmp_path / "v23", tmp_path / "current"
    _git_tree(TAG, old_root)
    new_root.mkdir()
    for name in TREE_FILES:
        src = REPO_ROOT / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, new_root / name)
    # v27: on a Blackwell GPU whose sageattn self-test passed (gpu_check.py info file), default jobs are unchanged
    gpu_ok = tmp_path / "gpu_sage_ok.json"
    gpu_ok.write_text(json.dumps({"name": "NVIDIA GeForce RTX 5090", "capability": "12.0", "total_gb": 31.36,
                                  "cuda": "12.8", "sageattn": "ok"}), encoding="utf-8")
    jobs = JOBS + [({"WORKER_GPU_INFO": str(gpu_ok)}, STUDIO),
                   ({"WORKER_GPU_INFO": str(gpu_ok), "ATTENTION_MODE": "sageattn"}, {**STUDIO, "width": 854})]
    old = _run(old_root, tmp_path, "old", jobs)
    new = _run(new_root, tmp_path, "new", jobs)
    assert len(old["prompts"]) == len(new["prompts"]) == len(jobs)
    for i, (a, b) in enumerate(zip(old["prompts"], new["prompts"])):
        assert a == b, f"job {i}: queued workflow differs"
    assert old["results"] == new["results"]
    assert all(set(r) == {"video"} for r in new["results"])
