# ==============================================================================
# tts_runner.py - Standalone Chatterbox Multilingual Inference Runner
# Executed strictly in /opt/tts-venv to guarantee complete ComfyUI isolation.
# ==============================================================================
import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [tts_runner] %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("tts_runner")

# chatterbox-tts 0.1.7: ChatterboxMultilingualTTS.from_local(ckpt_dir, device) loads exactly these files
# (mtl_tts.py lines 174, 179, 187, 192); the T3 checkpoint name is hard-coded there and cannot be selected.
T3_CHECKPOINT = "t3_mtl23ls_v2.safetensors"
REQUIRED_MODEL_FILES = ("ve.pt", T3_CHECKPOINT, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json")


def validate_model_dir(model_dir: str, expected_model: str) -> Path:
    """Checks (without importing torch) that the baked weights match what chatterbox 0.1.7 will load."""
    if expected_model != T3_CHECKPOINT:
        raise ValueError(
            f"Unsupported TTS model '{expected_model}'. chatterbox-tts 0.1.7 only loads {T3_CHECKPOINT}."
        )
    model_dir_path = Path(model_dir)
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"❌ Baked TTS weights missing in {model_dir}: {', '.join(missing)}")
    return model_dir_path


def free_comfyui_memory(timeout_sec: int = 20, min_free_gb: float = 4.5):
    """
    Checks GPU memory. If free VRAM is below min_free_gb, invokes ComfyUI /free
    endpoint and polls torch.cuda.mem_get_info() every 1s for up to timeout_sec.
    """
    import torch
    if not torch.cuda.is_available():
        logger.info("CUDA is not available; skipping ComfyUI VRAM check.")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    logger.info(f"Current GPU VRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total")

    if free_gb >= min_free_gb:
        logger.info(f"Sufficient VRAM ({free_gb:.2f} GB >= {min_free_gb} GB). No ComfyUI /free needed.")
        return

    logger.warning(
        f"Low free VRAM ({free_gb:.2f} GB < {min_free_gb} GB). "
        "Triggering ComfyUI POST /free to release loaded video models. "
        "NOTE: Subsequent video jobs will need to reload models, causing initial latency."
    )

    req = urllib.request.Request(
        "http://127.0.0.1:8188/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info(f"ComfyUI /free response status: {resp.status}")
    except Exception as e:
        logger.warning(f"Could not reach ComfyUI /free endpoint: {e}. Proceeding anyway.")

    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        time.sleep(1.0)
        torch.cuda.empty_cache()
        curr_free_bytes, _ = torch.cuda.mem_get_info()
        curr_free_gb = curr_free_bytes / (1024 ** 3)
        elapsed = time.time() - start_time
        logger.info(f"Polling VRAM: {curr_free_gb:.2f} GB free after {elapsed:.1f}s")
        if curr_free_gb >= min_free_gb:
            logger.info(f"✅ VRAM freed successfully: {curr_free_gb:.2f} GB available (was {free_gb:.2f} GB).")
            return

    logger.warning(f"Timed out after {timeout_sec}s waiting for {min_free_gb} GB VRAM. Proceeding with best effort.")


def preprocess_reference_audio(input_wav: str, output_wav: str, min_sec: float = 3.0, max_sec: float = 15.0):
    """
    Converts reference audio to mono 24kHz, trims silence, validates minimum length (>=3s),
    and clamps to maximum length (<=15s).
    """
    import librosa
    import soundfile as sf
    import numpy as np

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Reference audio not found: {input_wav}")

    y, sr = librosa.load(input_wav, sr=24000, mono=True)
    if y.size == 0:
        raise ValueError("Reference audio file is empty.")

    # Trim leading and trailing silence (top_db=25)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if y_trimmed.size == 0:
        y_trimmed = y

    duration = len(y_trimmed) / sr
    logger.info(f"Reference audio duration after silence trim: {duration:.2f}s")

    if duration < min_sec:
        raise ValueError(
            f"Reference audio is too short ({duration:.2f}s). "
            f"Please provide a reference audio of at least {min_sec:.1f} seconds."
        )

    if duration > max_sec:
        logger.info(f"Clamping reference audio from {duration:.2f}s to {max_sec:.1f}s")
        y_trimmed = y_trimmed[: int(max_sec * sr)]

    sf.write(output_wav, y_trimmed, sr, subtype="PCM_16")
    return output_wav


def split_text_into_chunks(text: str, max_chars: int = 250) -> list[str]:
    """
    Splits input text into sentence chunks respecting English and Hindi sentence boundaries
    (such as Devanagari danda '।' and '॥', full stops, question marks, exclamation marks).
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Regex for English & Hindi sentence delimiters
    # Includes Devanagari danda (\u0964), double danda (\u0965), ., ?, !, \n
    sentence_regex = re.compile(r"([^।॥.!?\n]+[।॥.!?\n]*|\n+)")
    raw_sentences = [s.strip() for s in sentence_regex.findall(cleaned) if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in raw_sentences:
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence is longer than max_chars, split by clauses/spaces
            if len(sentence) > max_chars:
                words = sentence.split(" ")
                sub_chunk = ""
                for w in words:
                    if len(sub_chunk) + len(w) + 1 <= max_chars:
                        sub_chunk = f"{sub_chunk} {w}".strip() if sub_chunk else w
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = w
                if sub_chunk:
                    chunks.append(sub_chunk)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_optimal_device() -> str:
    """
    Determines if CUDA is available AND if the device architecture is supported
    by the installed PyTorch build (e.g. sm_89, sm_80 vs sm_120).
    Falls back to 'cpu' if unsupported to avoid 'no kernel image' crashes.
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability()
        arch_name = f"sm_{major}{minor}"
        raw_arches = torch.cuda.get_arch_list() or (
            torch._C._cuda_getArchFlags().split() if hasattr(torch._C, "_cuda_getArchFlags") else []
        )
        if raw_arches and not any(arch_name in a for a in raw_arches):
            logger.warning(
                f"GPU architecture '{arch_name}' (Compute {major}.{minor}) is not supported "
                f"by this PyTorch build ({torch.__version__}, supported: {raw_arches}). "
                f"Falling back to CPU inference."
            )
            return "cpu"
    except Exception as e:
        logger.warning(f"Could not verify CUDA device architecture: {e}")
        return "cpu"
    return "cuda"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chatterbox Multilingual TTS Runner")
    parser.add_argument("--text", type=str, required=True, help="Input text prompt to synthesize")
    parser.add_argument("--audio-prompt", type=str, required=True, help="Path to reference audio clip")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save output WAV")
    parser.add_argument("--language-id", type=str, default="en", help="Target language code (e.g., 'en', 'hi')")
    parser.add_argument("--model-dir", type=str, default="/opt/tts-models", help="Directory containing baked weights")
    parser.add_argument("--expected-model", type=str, default=T3_CHECKPOINT, help="Expected T3 safetensors file")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Classifier-Free Guidance weight")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--min-vram-gb", type=float, default=4.5, help="Minimum free VRAM before invoking /free")
    args = parser.parse_args(argv)

    # 1. Validate Model Weights in Local Model Directory (before any heavy import)
    model_dir_path = validate_model_dir(args.model_dir, args.expected_model)

    import torch
    import numpy as np
    import soundfile as sf

    # Chatterbox uses perth.PerthImplicitWatermarker, which requires pkg_resources.
    # Fallback to DummyWatermarker if PerthImplicitWatermarker is None so generation never crashes.
    try:
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = getattr(perth, "DummyWatermarker", None)
    except Exception:
        pass

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    # 2. Validate Language
    lang = args.language_id.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language_id '{lang}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
        )

    # 3. Determine device capability BEFORE making any CUDA calls
    device = get_optimal_device()
    logger.info(f"Chatterbox target execution device: {device}")

    # 4. Check GPU VRAM and release ComfyUI memory only if running on GPU
    if device == "cuda":
        free_comfyui_memory(timeout_sec=20, min_free_gb=args.min_vram_gb)
    else:
        cpu_threads = max(1, min(os.cpu_count() or 4, 8))
        torch.set_num_threads(cpu_threads)
        logger.info(f"Configured CPU execution with {cpu_threads} PyTorch threads")

    # 5. Set Seeds for Deterministic Generation
    torch.manual_seed(args.seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 6. Preprocess Reference Audio
    temp_ref_path = str(model_dir_path / f"temp_ref_{os.getpid()}.wav")
    try:
        processed_ref_wav = preprocess_reference_audio(args.audio_prompt, temp_ref_path)

        # 7. Split Text into Chunks (Hindi + English aware)
        chunks = split_text_into_chunks(args.text, max_chars=250)
        if not chunks:
            raise ValueError("Input text prompt is empty or contains only whitespace.")

        logger.info(f"Synthesizing {len(chunks)} chunk(s) for language '{lang}'...")

        # 8. Load Chatterbox Multilingual Model locally (Offline). 0.1.7 signature: from_local(ckpt_dir, device)
        try:
            logger.info(f"Loading Chatterbox model on {device}...")
            model = ChatterboxMultilingualTTS.from_local(ckpt_dir=args.model_dir, device=device)
        except Exception as e:
            if device != "cpu" and ("no kernel image is available" in str(e) or "CUDA error" in str(e) or "not compatible" in str(e)):
                logger.warning(f"CUDA execution failed on {device}: {e}. Retrying with CPU fallback...")
                device = "cpu"
                torch.cuda.empty_cache()
                model = ChatterboxMultilingualTTS.from_local(ckpt_dir=args.model_dir, device="cpu")
            else:
                raise

        audio_segments = []
        silence_gap = np.zeros(int(0.15 * model.sr), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {i + 1}/{len(chunks)}: '{chunk[:60]}...'")
            try:
                wav_tensor = model.generate(
                    text=chunk,
                    language_id=lang,
                    audio_prompt_path=processed_ref_wav,
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                )
            except RuntimeError as e:
                if ("no kernel image is available" in str(e) or "CUDA error" in str(e)) and device != "cpu":
                    logger.warning(f"CUDA error during generation: {e}. Switching to CPU...")
                    device = "cpu"
                    torch.cuda.empty_cache()
                    model = ChatterboxMultilingualTTS.from_local(ckpt_dir=args.model_dir, device="cpu")
                    wav_tensor = model.generate(
                        text=chunk,
                        language_id=lang,
                        audio_prompt_path=processed_ref_wav,
                        exaggeration=args.exaggeration,
                        cfg_weight=args.cfg_weight,
                        temperature=args.temperature,
                    )
                else:
                    raise
            wav_np = wav_tensor.detach().cpu().numpy().squeeze()
            audio_segments.append(wav_np)
            if i < len(chunks) - 1:
                audio_segments.append(silence_gap)

        combined_audio = np.concatenate(audio_segments)

        # 8. Peak Normalization (-1.0 dBFS)
        max_amplitude = np.max(np.abs(combined_audio))
        if max_amplitude > 0:
            target_amplitude = 10.0 ** (-1.0 / 20.0)
            combined_audio = combined_audio * (target_amplitude / max_amplitude)

        # 9. Save as 16-bit PCM WAV
        out_dir = os.path.dirname(os.path.abspath(args.output_path))
        os.makedirs(out_dir, exist_ok=True)
        sf.write(args.output_path, combined_audio, model.sr, subtype="PCM_16")

        duration_sec = float(len(combined_audio) / model.sr)
        logger.info(f"✅ Generated audio saved to {args.output_path} ({duration_sec:.2f}s, {model.sr}Hz)")

        # Output machine-readable JSON result to stdout for tts_handler
        result = {
            "output_path": args.output_path,
            "sample_rate": model.sr,
            "duration_sec": duration_sec,
            "chunks": len(chunks),
        }
        print(json.dumps(result))

    finally:
        if os.path.exists(temp_ref_path):
            try:
                os.remove(temp_ref_path)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"TTS generation error: {e}", exc_info=True)
        sys.exit(1)
