# ==============================================================================
# tts_handler.py - RunPod Serverless Handler for Chatterbox Voice Cloning
# ==============================================================================
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

logger = logging.getLogger("tts_handler")
logger.setLevel(logging.INFO)


def _resolve_tts_python() -> str:
    """Returns /opt/tts-venv/bin/python in Docker, or sys.executable for local unit testing."""
    venv_python = "/opt/tts-venv/bin/python"
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def handle_tts(job_input: dict) -> dict:
    """
    Handles task == "tts" requests in an isolated subprocess.
    Returns:
        {
            "audio_base64": "<base64_str>",
            "sample_rate": 24000,
            "duration_sec": 12.34,
            "chunks": 2
        }
    or on failure:
        {"error": "<error description>"}
    """
    task_id = f"tts_{uuid.uuid4().hex[:8]}"
    work_dir = Path(f"/tmp/tts/{task_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Validate Text Input
        text = job_input.get("text", "").strip()
        if not text:
            return {"error": "Missing required field: text prompt cannot be empty."}

        # 2. Extract Reference Audio (base64 only; URL and path inputs are rejected)
        if job_input.get("reference_audio_url") or job_input.get("reference_audio_path"):
            return {"error": "reference_audio_url and reference_audio_path are not supported. Provide reference_audio_base64."}

        ref_path = work_dir / "ref_input.wav"
        if job_input.get("reference_audio_base64"):
            try:
                b64_data = job_input["reference_audio_base64"]
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                with open(ref_path, "wb") as f:
                    f.write(base64.b64decode(b64_data, validate=True))
            except Exception as e:
                return {"error": f"Failed to decode reference_audio_base64: {e}"}
        else:
            return {"error": "Missing reference audio. Provide reference_audio_base64."}

        # 3. Parameters
        language_id = job_input.get("language_id", "en")
        temperature = float(job_input.get("temperature", 0.8))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))
        exaggeration = float(job_input.get("exaggeration", 0.5))
        seed = int(job_input.get("seed", 42))
        model_dir = os.getenv("TTS_MODEL_DIR", "/opt/tts-models")
        expected_model = os.getenv("TTS_MODEL", "t3_mtl23ls_v2.safetensors")

        output_path = work_dir / "output.wav"
        tts_python = _resolve_tts_python()

        # 4. Formulate Subprocess Execution
        cmd = [
            tts_python,
            "-m", "tts.tts_runner",
            "--text", text,
            "--audio-prompt", str(ref_path),
            "--output-path", str(output_path),
            "--language-id", language_id,
            "--model-dir", model_dir,
            "--expected-model", expected_model,
            "--temperature", str(temperature),
            "--cfg-weight", str(cfg_weight),
            "--exaggeration", str(exaggeration),
            "--seed", str(seed),
        ]

        clean_env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": model_dir,
            "PYTHONPATH": f"/tts:{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}:" + os.environ.get("PYTHONPATH", "")
        }

        logger.info(f"[{task_id}] Launching isolated Chatterbox runner: {' '.join(cmd[:6])}...")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=clean_env,
            timeout=600
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.strip()
            # If stderr contains a python traceback, prioritize the traceback lines over general warnings
            lines = [l for l in stderr_text.splitlines() if l.strip()]
            tb_lines = []
            capturing = False
            for l in lines:
                if "Traceback (most recent call last):" in l:
                    capturing = True
                if capturing:
                    tb_lines.append(l)
            err_msg = "\n".join(tb_lines) if tb_lines else (stderr_text or proc.stdout.strip() or f"Process exited with code {proc.returncode}")
            logger.error(f"[{task_id}] TTS Runner execution failed: {err_msg}")
            return {"error": f"TTS generation failed: {err_msg}"}

        # Parse JSON output from runner stdout
        runner_stdout = proc.stdout.strip()
        last_line = runner_stdout.split("\n")[-1] if runner_stdout else "{}"
        try:
            runner_meta = json.loads(last_line)
        except Exception:
            runner_meta = {}

        if not output_path.exists():
            return {"error": f"Generated audio file missing: {output_path}. Runner logs: {runner_stdout}"}

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        duration_sec = runner_meta.get("duration_sec", 0.0)
        sample_rate = runner_meta.get("sample_rate", 24000)
        chunks_count = runner_meta.get("chunks", 1)

        logger.info(f"[{task_id}] ✅ TTS generation successful: {duration_sec:.2f}s, {chunks_count} chunks")
        return {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "chunks": chunks_count,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"[{task_id}] TTS execution timed out after 600s")
        return {"error": "TTS generation timed out after 600 seconds."}
    except Exception as e:
        logger.error(f"[{task_id}] Unexpected error in handle_tts: {e}", exc_info=True)
        return {"error": f"Internal TTS handler error: {str(e)}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
