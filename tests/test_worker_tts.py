# ==============================================================================
# test_worker_tts.py - pytest suite for the RunPod worker TTS pipeline
# Runs on Windows without a GPU: torch, chatterbox, runpod, librosa and soundfile are mocked.
# ==============================================================================
import ast
import base64
import difflib
import json
import re
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import REPO_ROOT
from tts import tts_runner
from tts.tts_runner import split_text_into_chunks, preprocess_reference_audio, validate_model_dir, T3_CHECKPOINT
from tts.tts_handler import handle_tts
import handler

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import patch_handler_tts  # noqa: E402

DUMMY_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
ANCHOR = '    job_input = job.get("input", {})'
EXPECTED_BLOCK = [
    "",
    '    if job_input.get("task") == "tts":',
    "        from tts.tts_handler import handle_tts",
    "        return handle_tts(job_input)",
]


def fake_runner(stdout_meta=None):
    """subprocess.run replacement that writes a dummy output.wav and prints runner JSON."""
    def run(cmd, **kwargs):
        out_file = Path(cmd[cmd.index("--output-path") + 1])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(DUMMY_WAV)
        res = MagicMock()
        res.returncode = 0
        res.stdout = json.dumps(stdout_meta or {"output_path": str(out_file), "sample_rate": 24000, "duration_sec": 4.5, "chunks": 1})
        res.stderr = ""
        return res
    return MagicMock(side_effect=run)


def make_model_dir(tmp_path, skip=()):
    model_dir = tmp_path / "tts-models"
    model_dir.mkdir()
    for name in ("ve.pt", T3_CHECKPOINT, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"):
        if name not in skip:
            (model_dir / name).write_bytes(b"x")
    return model_dir


@pytest.fixture(scope="module")
def head_handler_source():
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "show", "HEAD:handler.py"], capture_output=True, check=True)
    except Exception:
        pytest.skip("git HEAD:handler.py not available")
    return out.stdout.decode("utf-8")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def read_text(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


# ---------------------------------------------------------------- text chunking
def test_empty_text():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\t  ") == []


def test_single_short_sentence():
    text = "Hello, this is a test."
    assert split_text_into_chunks(text, max_chars=100) == [text]


def test_hindi_danda_splitting():
    hindi_text = (
        "नमस्ते, यह चैटर्बॉक्स टेक्स्ट-टू-स्पीच का परीक्षण है। "
        "यह दूसरी पंक्ति है जिसमें आवाज़ की प्रतिलिपि बनाई जाएगी। "
        "और यह अंतिम वाक्य है जो पूरा होता है।"
    )
    chunks = split_text_into_chunks(hindi_text, max_chars=60)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 65 for chunk in chunks)


def test_long_sentence_splitting():
    chunks = split_text_into_chunks("This is a word " * 40, max_chars=200)  # 600 characters without punctuation
    assert len(chunks) > 1
    assert all(len(chunk) <= 205 for chunk in chunks)


# ---------------------------------------------------------------- reference preprocessing
def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_reference_audio(str(tmp_path / "nonexistent.wav"), str(tmp_path / "out.wav"))


def test_short_audio_rejected(tmp_path):
    audio = np.ones(48000, dtype=np.float32)  # 2 seconds at 24kHz
    ref = tmp_path / "ref.wav"
    ref.touch()
    with patch("librosa.load", return_value=(audio, 24000)), patch("librosa.effects.trim", return_value=(audio, (0, len(audio)))):
        with pytest.raises(ValueError, match="too short"):
            preprocess_reference_audio(str(ref), str(tmp_path / "out.wav"), min_sec=3.0)


def test_long_audio_clamped(tmp_path):
    audio = np.ones(24000 * 20, dtype=np.float32)  # 20 seconds at 24kHz
    ref, out = tmp_path / "ref.wav", tmp_path / "out.wav"
    ref.touch()
    with patch("librosa.load", return_value=(audio, 24000)), \
            patch("librosa.effects.trim", return_value=(audio, (0, len(audio)))), \
            patch("soundfile.write") as mock_write:
        result = preprocess_reference_audio(str(ref), str(out), min_sec=3.0, max_sec=15.0)
    assert result == str(out)
    mock_write.assert_called_once()
    assert len(mock_write.call_args[0][1]) == 24000 * 15


# ---------------------------------------------------------------- handler validation
def test_missing_text():
    res = handle_tts({"reference_audio_base64": "AAAA"})
    assert "text prompt" in res["error"]


def test_missing_reference_audio():
    res = handle_tts({"text": "Hello world"})
    assert "reference audio" in res["error"].lower()


def test_invalid_base64():
    res = handle_tts({"text": "Hello world", "reference_audio_base64": "!!!not_valid_b64!!!"})
    assert "error" in res


def test_successful_runner_mock_uses_v2_checkpoint_offline():
    run = fake_runner()
    with patch("subprocess.run", run):
        res = handle_tts({
            "task": "tts",
            "text": "Valid test prompt",
            "reference_audio_base64": base64.b64encode(DUMMY_WAV).decode("utf-8"),
            "language_id": "en",
        })
    assert "error" not in res
    assert base64.b64decode(res["audio_base64"]) == DUMMY_WAV
    assert (res["sample_rate"], res["duration_sec"], res["chunks"]) == (24000, 4.5, 1)

    cmd, kwargs = run.call_args[0][0], run.call_args[1]
    assert cmd[cmd.index("--expected-model") + 1] == T3_CHECKPOINT
    assert cmd[cmd.index("--language-id") + 1] == "en"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


# ---------------------------------------------------------------- reference source restriction (url/path removed)
def test_url_only_rejected():
    with patch("urllib.request.urlretrieve") as mock_urlretrieve, patch("shutil.copy2") as mock_copy2:
        res = handle_tts({"text": "Hello world", "reference_audio_url": "http://169.254.169.254/latest/meta-data"})
    assert "not supported" in res["error"]
    mock_urlretrieve.assert_not_called()
    mock_copy2.assert_not_called()


def test_path_only_rejected():
    with patch("urllib.request.urlretrieve") as mock_urlretrieve, patch("shutil.copy2") as mock_copy2:
        # Point at a file that really exists to prove existence no longer matters
        res = handle_tts({"text": "Hello world", "reference_audio_path": str(Path(__file__).resolve())})
    assert "not supported" in res["error"]
    mock_urlretrieve.assert_not_called()
    mock_copy2.assert_not_called()


def test_base64_plus_url_rejected():
    with patch("urllib.request.urlretrieve") as mock_urlretrieve, patch("subprocess.run") as mock_run:
        res = handle_tts({
            "text": "Hello world",
            "reference_audio_base64": base64.b64encode(b"RIFF0000WAVE").decode(),
            "reference_audio_url": "file:///etc/passwd",
        })
    assert "not supported" in res["error"]
    mock_urlretrieve.assert_not_called()
    mock_run.assert_not_called()


def test_missing_reference_mentions_base64_only():
    res = handle_tts({"text": "Hello world"})
    assert "reference_audio_base64" in res["error"]
    assert "reference_audio_url" not in res["error"]


# ---------------------------------------------------------------- tts_runner: chatterbox 0.1.7 checkpoint + loading API
def test_t3_checkpoint_is_the_0_1_7_file():
    assert T3_CHECKPOINT == "t3_mtl23ls_v2.safetensors"


def test_validate_model_dir_accepts_complete_dir(tmp_path):
    model_dir = make_model_dir(tmp_path)
    assert validate_model_dir(str(model_dir), T3_CHECKPOINT) == model_dir


@pytest.mark.parametrize("missing", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"])
def test_validate_model_dir_reports_missing_file(tmp_path, missing):
    model_dir = make_model_dir(tmp_path, skip=(missing,))
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        validate_model_dir(str(model_dir), T3_CHECKPOINT)


def test_validate_model_dir_rejects_other_checkpoint_names(tmp_path):
    model_dir = make_model_dir(tmp_path)
    with pytest.raises(ValueError, match="only loads t3_mtl23ls_v2.safetensors"):
        validate_model_dir(str(model_dir), "some_other_model.safetensors")


def test_runner_main_uses_0_1_7_from_local_signature(tmp_path, monkeypatch, capsys):
    model_dir = make_model_dir(tmp_path)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(DUMMY_WAV)

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_cls = MagicMock()
    model = fake_cls.from_local.return_value
    model.sr = 24000
    model.generate.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(2400, dtype=np.float32)
    fake_mtl = types.ModuleType("chatterbox.mtl_tts")
    fake_mtl.ChatterboxMultilingualTTS = fake_cls
    fake_mtl.SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", fake_mtl)
    monkeypatch.setattr(tts_runner, "free_comfyui_memory", lambda **kwargs: None)
    monkeypatch.setattr(tts_runner, "preprocess_reference_audio", lambda src, dst: dst)

    tts_runner.main([
        "--text", "Hello there.", "--audio-prompt", str(ref), "--output-path", str(tmp_path / "out.wav"),
        "--language-id", "en", "--model-dir", str(model_dir),
    ])

    fake_cls.from_local.assert_called_once_with(ckpt_dir=str(model_dir), device="cpu")
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (result["sample_rate"], result["chunks"], result["duration_sec"]) == (24000, 1, 0.1)


def test_runner_main_fails_before_loading_torch_when_weights_missing(tmp_path, monkeypatch):
    model_dir = make_model_dir(tmp_path, skip=("s3gen.pt",))
    monkeypatch.setitem(sys.modules, "torch", None)  # any torch import would raise ImportError
    with pytest.raises(FileNotFoundError, match="s3gen.pt"):
        tts_runner.main(["--text", "x", "--audio-prompt", "a.wav", "--output-path", "o.wav", "--model-dir", str(model_dir)])


# ---------------------------------------------------------------- Dockerfile.tts guards (static checks)
@pytest.fixture(scope="module")
def dockerfile_tts():
    return (REPO_ROOT / "Dockerfile.tts").read_text(encoding="utf-8")


def test_dockerfile_pins_hf_revision_to_commit_hash(dockerfile_tts):
    match = re.search(r'^ARG HF_REVISION="([0-9a-f]+)"$', dockerfile_tts, re.M)
    assert match and len(match.group(1)) == 40
    assert "revision='${HF_REVISION}'" in dockerfile_tts


def test_dockerfile_uses_v2_checkpoint_and_verifies_it(dockerfile_tts):
    assert 'ARG TTS_MODEL="t3_mtl23ls_v2.safetensors"' in dockerfile_tts
    assert "inspect.getsource(m.ChatterboxMultilingualTTS.from_local)" in dockerfile_tts
    assert dockerfile_tts.count("  t3_mtl23ls_v2.safetensors\"") == 1
    assert "sha256sum -c -" in dockerfile_tts


def test_dockerfile_never_copies_local_handler(dockerfile_tts):
    assert not re.search(r"^COPY\s+handler\.py", dockerfile_tts, re.M)
    assert "patch_handler_tts.py /handler.py" in dockerfile_tts


def test_dockerfile_freeze_diff_is_last_run_step(dockerfile_tts):
    run_lines = [line for line in dockerfile_tts.splitlines() if line.startswith("RUN ")]
    assert "comfy_freeze_post" in run_lines[-1]


def test_no_v3_or_removed_api_references_anywhere():
    files = list((REPO_ROOT / "tts").glob("*.py")) + [
        REPO_ROOT / "Dockerfile.tts", REPO_ROOT / "test_input_tts.json", REPO_ROOT / "scripts" / "patch_handler_tts.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"v3|_resolve_multilingual_t3_model|t3_model\s*=", text, re.I), path.name


def test_handler_default_model_matches_runner():
    source = (REPO_ROOT / "tts" / "tts_handler.py").read_text(encoding="utf-8")
    assert f'os.getenv("TTS_MODEL", "{T3_CHECKPOINT}")' in source


# ---------------------------------------------------------------- scripts/patch_handler_tts.py
def test_head_handler_gets_exactly_four_added_lines(tmp_path, head_handler_source):
    path = write_text(tmp_path / "handler.py", head_handler_source)
    assert patch_handler_tts.main(["patch", str(path)]) == 0
    patched = read_text(path)
    ast.parse(patched)
    diff = list(difflib.unified_diff(head_handler_source.splitlines(), patched.splitlines(), lineterm="", n=0))
    added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++")]
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    assert removed == []
    # difflib may attribute the duplicated blank line to either side, so compare as a multiset
    assert Counter(added) == Counter(EXPECTED_BLOCK)
    # Exact layout: anchor, blank, 3-line block, then the original following lines unchanged
    lines = patched.splitlines()
    anchor = lines.index(ANCHOR)
    assert lines[anchor + 1:anchor + 5] == EXPECTED_BLOCK
    assert lines[:anchor + 1] + lines[anchor + 5:] == head_handler_source.splitlines()


def test_crlf_line_endings_preserved(tmp_path, head_handler_source):
    crlf_source = head_handler_source.replace("\r\n", "\n").replace("\n", "\r\n")
    path = write_text(tmp_path / "handler.py", crlf_source)
    assert patch_handler_tts.main(["patch", str(path)]) == 0
    patched = read_text(path)
    assert "\n" not in patched.replace("\r\n", "")
    assert patched.count("\r\n") == crlf_source.count("\r\n") + 4


def test_patched_handler_routes_tts(head_handler_source):
    assert 'if job_input.get("task") == "tts":' in patch_handler_tts.patch_source(head_handler_source)


def test_local_handler_already_patched_fails(tmp_path):
    source = read_text(REPO_ROOT / "handler.py")
    path = write_text(tmp_path / "handler.py", source)
    assert patch_handler_tts.main(["patch", str(path)]) == 1
    assert read_text(path) == source


def test_zero_anchor_fails_untouched(tmp_path):
    source = "def handler(job):\n    return {}\n"
    path = write_text(tmp_path / "handler.py", source)
    assert patch_handler_tts.main(["patch", str(path)]) == 1
    assert read_text(path) == source


def test_two_anchors_fail_untouched(tmp_path):
    source = (
        "def a(job):\n    job_input = job.get(\"input\", {})\n    return job_input\n\n"
        "def b(job):\n    job_input = job.get(\"input\", {})\n    return job_input\n"
    )
    path = write_text(tmp_path / "handler.py", source)
    assert patch_handler_tts.main(["patch", str(path)]) == 1
    assert read_text(path) == source


# ---------------------------------------------------------------- handler.py routing (local working copy)
def test_task_tts_routes_to_tts_handler():
    with patch("tts.tts_handler.handle_tts", return_value={"audio_base64": "dummy", "duration_sec": 3.0}) as mock_handle_tts:
        job = {"input": {"task": "tts", "text": "hello"}}
        res = handler.handler(job)
    mock_handle_tts.assert_called_once_with(job["input"])
    assert res == {"audio_base64": "dummy", "duration_sec": 3.0}


def test_cmd_diagnostic_task_still_works():
    res = handler.handler({"input": {"cmd": "echo healthcheck"}})
    assert res.get("cmd") == "echo healthcheck"
    assert "healthcheck" in res.get("output", "")


def test_video_job_does_not_route_to_tts():
    # A video job with no task key should proceed past the tts check; process_input aborts it safely
    with patch("handler.get_workflow_path", return_value="/I2V_single.json"), \
            patch("handler.process_input", side_effect=Exception("Intentionally stopping at video workflow pipeline")):
        with pytest.raises(Exception, match="Intentionally stopping at video workflow pipeline"):
            handler.handler({"input": {"image_url": "http://example.com/test.jpg"}})
