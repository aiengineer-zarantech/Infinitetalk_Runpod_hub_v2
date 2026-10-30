# ==============================================================================
# scripts/patch_handler_guard.py (build-time patch used by Dockerfile.v18) and the
# local handler.py: exactly 2 added lines, nothing else.
# ==============================================================================
import ast
import difflib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import patch_handler_guard  # noqa: E402

BLOCK = patch_handler_guard.BLOCK
ANCHOR = patch_handler_guard.ANCHOR


@pytest.fixture(scope="module")
def head_handler_source():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "show", "HEAD:handler.py"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git HEAD:handler.py not available")
    return out.stdout.decode("utf-8").replace("\r\n", "\n")


def read(path):
    return path.read_bytes().decode("utf-8")


def write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


def added_removed(before, after):
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0))
    added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++")]
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    return added, removed


def tts_patched(source):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import patch_handler_tts
    return patch_handler_tts.patch_source(source)


@pytest.mark.parametrize("with_tts_block", [False, True])
def test_head_handler_gets_exactly_two_lines_before_start(tmp_path, head_handler_source, with_tts_block):
    # v13+ images have the TTS block already, so test both base shapes
    source = tts_patched(head_handler_source) if with_tts_block else head_handler_source
    path = write(tmp_path / "handler.py", source)
    assert patch_handler_guard.main(["patch", str(path)]) == 0
    patched = read(path)
    ast.parse(patched)
    added, removed = added_removed(source, patched)
    assert removed == [] and added == BLOCK
    lines = patched.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(ANCHOR))
    assert lines[start - 2:start] == BLOCK


def test_crlf_preserved(tmp_path, head_handler_source):
    crlf = head_handler_source.replace("\n", "\r\n")
    path = write(tmp_path / "handler.py", crlf)
    assert patch_handler_guard.main(["patch", str(path)]) == 0
    patched = read(path)
    assert "\n" not in patched.replace("\r\n", "")
    assert patched.count("\r\n") == crlf.count("\r\n") + 2


def test_already_patched_fails_untouched(tmp_path):
    source = read(REPO_ROOT / "handler.py")
    path = write(tmp_path / "handler.py", source)
    assert patch_handler_guard.main(["patch", str(path)]) == 1
    assert read(path) == source


@pytest.mark.parametrize("source", [
    "def handler(job):\n    return {}\n\ndef get_videos(ws, p):\n    return {}\n",  # no anchor
    "def handler(job):\n    return {}\n\ndef get_videos(ws, p):\n    return {}\n\n"
    "runpod.serverless.start({})\nrunpod.serverless.start({})\n",  # two anchors
    "def handler(job):\n    return {}\n\nrunpod.serverless.start({})\n",  # no get_videos
    "def handler(job):\n    runpod.serverless.start({})\n\ndef get_videos(ws, p):\n    return {}\n",  # indented only
])
def test_bad_sources_fail_untouched(tmp_path, source):
    path = write(tmp_path / "handler.py", source)
    assert patch_handler_guard.main(["patch", str(path)]) == 1
    assert read(path) == source


# v23: offload fail-safe in queue_prompt + opt-in vae_tiling job input (TESTING.md step 2c)
V23_QUEUE_PROMPT = [
    "    # Offload fail-safe right before /prompt (also with WORKER_GUARD_ENABLED=0); logs \"[OFFLOAD] final: ...\"",
    "    from worker_guard.job_guard import final_offload_check",
    "    final_offload_check(prompt)",
    "",
]
V23_VAE_TILING = [
    "        elif job_input.get(\"vae_tiling\") and ctype in (\"WanVideoImageToVideoMultiTalk\", \"WanVideoDecode\", \"WanVideoEncode\"):",
    "            # Test input (TESTING.md step 2c): tiled VAE for the per-window decode/encode and the final decode",
    "            inp = node_data.setdefault(\"inputs\", {})",
    "            inp[\"tiled_vae\" if ctype == \"WanVideoImageToVideoMultiTalk\" else \"enable_vae_tiling\"] = True",
    "            logger.info(f\"✅ 노드 {node_id} ({ctype}) VAE tiling on\")",
]


def test_local_handler_differs_from_baseline_by_exactly_the_block_and_v23_lines():
    baseline = Path("D:/Serverless2/.baselines/guard_phase_handler.py.before")
    if not baseline.exists():
        pytest.skip("phase baseline not available on this machine")
    added, removed = added_removed(read(baseline), read(REPO_ROOT / "handler.py"))
    assert removed == [] and added == V23_QUEUE_PROMPT + V23_VAE_TILING + BLOCK


def test_dockerfile_v18_never_copies_local_handler():
    text = (REPO_ROOT / "Dockerfile.v18").read_text(encoding="utf-8")
    assert "FROM digamberjhazaran/infinitetalk-runpod:v17" in text
    assert "COPY handler.py" not in text and "COPY . ." not in text
    assert "patch_handler_guard.py /handler.py" in text
    assert "pip freeze" in text
    for forbidden in ("pip install", "--cpu", "I2V_", "V2V_"):
        assert forbidden not in text, forbidden


def test_dockerfile_v23_replaces_only_worker_guard_and_the_checked_handler():
    text = (REPO_ROOT / "Dockerfile.v23").read_text(encoding="utf-8")
    assert "FROM digamberjhazaran/infinitetalk-runpod:v22" in text
    copies = [line.split(None, 1)[1] for line in text.splitlines() if line.startswith("COPY ")]
    assert copies == ["worker_guard/ /worker_guard/", "handler.py /handler.py"]
    # v22's handler must be the exact file the local handler.py was edited from
    assert "5534acefa8ad762873b273051ee2dcbccea29a7e8bd2f6721757558accb15b8c  /handler.py" in text
    fingerprinted = text.split("> /tmp/unchanged.sha256", 1)[0].rsplit("sha256sum", 1)[1]
    for name in ("/entrypoint.sh", "/setup_models.py", "/I2V_single.json", "/I2V_multi.json", "/V2V_single.json",
                 "/V2V_multi.json", "/I2V_single_compiled.json", "/I2V_multi_compiled.json"):
        assert name in fingerprinted, name
    assert "pip install" not in text and "--cpu" not in text and "sha256sum -c --quiet /tmp/unchanged.sha256" in text


def test_dockerfile_v21_only_replaces_worker_guard():
    text = (REPO_ROOT / "Dockerfile.v21").read_text(encoding="utf-8")
    assert "FROM digamberjhazaran/infinitetalk-runpod:v20" in text
    copies = [line.split(None, 1)[1] for line in text.splitlines() if line.startswith("COPY ")]
    assert copies == ["worker_guard/ /worker_guard/"]
    fingerprinted = text.split("sha256sum", 1)[1].split("> /tmp/unchanged.sha256", 1)[0]
    for name in ("/handler.py", "/entrypoint.sh", "/setup_models.py", "/I2V_single.json", "/I2V_multi.json",
                 "/V2V_single.json", "/V2V_multi.json", "/I2V_single_compiled.json", "/I2V_multi_compiled.json"):
        assert name in fingerprinted, name
    assert "pip install" not in text and "--cpu" not in text and "sha256sum -c /tmp/unchanged.sha256" in text


def test_dockerfile_v20_adds_only_guard_and_compiled_copies():
    text = (REPO_ROOT / "Dockerfile.v20").read_text(encoding="utf-8")
    assert "FROM digamberjhazaran/infinitetalk-runpod:v19" in text
    copies = [line.split(None, 1)[1] for line in text.splitlines() if line.startswith("COPY ")]
    assert copies == ["worker_guard/ /worker_guard/", "I2V_single_compiled.json I2V_multi_compiled.json /",
                      "scripts/verify_wrapper_contract.py /tmp/verify_wrapper_contract.py"]
    assert "verify_wrapper_contract.py /ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper" in text
    for name in ("/I2V_single.json", "/I2V_multi.json", "/V2V_single.json", "/V2V_multi.json", "/handler.py"):
        assert name in text.split("sha256sum", 1)[1].split("\n", 1)[0]
    assert "pip install" not in text and "--cpu" not in text and "sha256sum -c /tmp/unchanged.sha256" in text


def test_dockerfile_v19_only_replaces_worker_guard():
    text = (REPO_ROOT / "Dockerfile.v19").read_text(encoding="utf-8")
    assert "FROM digamberjhazaran/infinitetalk-runpod:v18" in text
    copies = [line.split()[1] for line in text.splitlines() if line.startswith("COPY ")]
    assert copies == ["worker_guard/"]
    assert "pip install" not in text and "--cpu" not in text
    assert "sha256sum -c /tmp/unchanged.sha256" in text and "pip freeze" in text
