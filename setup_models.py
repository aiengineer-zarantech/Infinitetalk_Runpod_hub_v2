#!/usr/bin/env python3
import os
import sys
import shutil
import time
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

COMFY_MODELS_DIR = "/ComfyUI/models"
LOCAL_MODELS_DIR = "/models_local"
VOLUME_ROOT = "/runpod-volume"
# Searched in order: <root>/models/<type>/<file>, <root>/<type>/<file>, <root>/<file>
SOURCE_ROOTS = [VOLUME_ROOT, "/workspace"]

MODEL_DESTINATIONS = {
    "Wan2_1_VAE_bf16.safetensors": "vae",
    "clip_vision_h.safetensors": "clip_vision",
    "umt5-xxl-enc-fp8_e4m3fn.safetensors": "text_encoders",
    "MelBandRoformer_fp16.safetensors": "diffusion_models",
    "wav2vec2-chinese-base_fp16.safetensors": "diffusion_models",
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors": "loras",
    "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors": "diffusion_models",
    "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors": "diffusion_models",
    "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors": "diffusion_models",
}

# Must exist on the volume at startup, but are copied on the first job that needs them
# (worker_guard.job_guard -> ensure_model), so they do not slow down cold start.
LAZY_MODELS = {
    "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors": "diffusion_models",
}

COPY_THREADS = int(os.environ.get("COPY_THREADS", "16"))
COPY_PART_SIZE = int(os.environ.get("COPY_PART_MB", "64")) * 1024 * 1024


def _clamp_file_workers(value):
    """Number of files copied at the same time, always 4..8."""
    try:
        return min(8, max(4, int(value)))
    except (TypeError, ValueError):
        return 6


COPY_FILE_WORKERS = _clamp_file_workers(os.environ.get("COPY_FILE_WORKERS", "6"))
# Caps parallel byte-range reads across ALL files (each range = separate S3 request on GeeseFS)
COPY_MAX_INFLIGHT = max(1, int(os.environ.get("COPY_MAX_INFLIGHT", "32")))

_inflight_ranges = threading.BoundedSemaphore(COPY_MAX_INFLIGHT)
_abort_copies = threading.Event()
_space_lock = threading.Lock()
_space_claims = {}  # "<local_path>.tmp" -> bytes that copy needs in total
_ensure_lock = threading.Lock()
_print_lock = threading.Lock()


class DiskSpaceError(Exception):
    pass


class CopyError(Exception):
    pass


def _log(msg, err=False):
    with _print_lock:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _copy_range(src_fd, dst_fd, offset, length):
    """Copy one byte range with pread/pwrite (each range = separate S3 request on GeeseFS)."""
    with _inflight_ranges:
        buffer_size = 8 * 1024 * 1024
        end = offset + length
        while offset < end:
            if _abort_copies.is_set():
                raise IOError("Copy aborted because another copy failed")
            to_read = min(buffer_size, end - offset)
            chunk = os.pread(src_fd, to_read, offset)
            if not chunk:
                raise IOError(f"Unexpected EOF at offset {offset}")
            written = 0
            while written < len(chunk):
                w = os.pwrite(dst_fd, chunk[written:], offset + written)
                if w <= 0:
                    raise IOError(f"Failed to write at offset {offset + written}")
                written += w
            offset += len(chunk)
    return length


def copy_file_chunked(src, dst):
    """Copy file in parallel byte ranges (COPY_THREADS x COPY_PART_MB), timing and calculating speed in MB/s."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp"
    if os.path.exists(tmp_dst):
        os.remove(tmp_dst)

    start_time = time.time()
    total_bytes = os.path.getsize(src)
    ranges = [
        (off, min(COPY_PART_SIZE, total_bytes - off))
        for off in range(0, total_bytes, COPY_PART_SIZE)
    ]
    src_fd = os.open(src, os.O_RDONLY)
    dst_fd = os.open(tmp_dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(dst_fd, total_bytes)
        with ThreadPoolExecutor(max_workers=max(1, min(COPY_THREADS, len(ranges)))) as pool:
            futures = [pool.submit(_copy_range, src_fd, dst_fd, off, ln) for off, ln in ranges]
            for fut in as_completed(futures):
                fut.result()
    finally:
        os.close(src_fd)
        os.close(dst_fd)
    if os.path.getsize(tmp_dst) != total_bytes:
        raise IOError(f"Size mismatch after copy: {os.path.getsize(tmp_dst)} != {total_bytes}")
    os.replace(tmp_dst, dst)
    elapsed = time.time() - start_time
    mb = total_bytes / (1024 * 1024)
    speed = (mb / elapsed) if elapsed > 0 else 0.0
    return mb, elapsed, speed


def candidate_paths(model_name, subfolder):
    paths = []
    for root in SOURCE_ROOTS:
        paths += [
            os.path.join(root, "models", subfolder, model_name),
            os.path.join(root, subfolder, model_name),
            os.path.join(root, model_name),
        ]
    return paths


def find_source(model_name, subfolder):
    for cand in candidate_paths(model_name, subfolder):
        try:
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
    return None


def model_paths(model_name, subfolder):
    """Returns (dest_path in /ComfyUI/models, local_path in /models_local), creating both folders."""
    dest_dir = os.path.join(COMFY_MODELS_DIR, subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    local_dir = os.path.join(LOCAL_MODELS_DIR, subfolder)
    os.makedirs(local_dir, exist_ok=True)
    return os.path.join(dest_dir, model_name), os.path.join(local_dir, model_name)


def local_copy_ok(local_path, size):
    return os.path.exists(local_path) and not os.path.islink(local_path) and os.path.getsize(local_path) == size


def link_is_valid(dest_path, local_path, size):
    """True if the ComfyUI path is a symlink to the local copy and the local copy has the expected size."""
    try:
        return (
            os.path.islink(dest_path)
            and os.path.realpath(dest_path) == os.path.realpath(local_path)
            and os.path.isfile(local_path)
            and os.path.getsize(local_path) == size
        )
    except OSError:
        return False


def _link(local_path, dest_path):
    if os.path.islink(dest_path) or os.path.exists(dest_path):
        os.remove(dest_path)
    os.symlink(local_path, dest_path)


def _allocated_bytes(path):
    try:
        st = os.stat(path)
    except OSError:
        return 0
    blocks = getattr(st, "st_blocks", None)
    return blocks * 512 if blocks is not None else 0


def _claim_space(model_name, local_path, nbytes):
    """Reserves space for one copy. Copies already running still need their unwritten bytes."""
    with _space_lock:
        pending = sum(max(0, size - _allocated_bytes(tmp)) for tmp, size in _space_claims.items())
        free = shutil.disk_usage(LOCAL_MODELS_DIR).free
        available = free - pending
        if available < nbytes:
            raise DiskSpaceError(
                f"❌ [DISK SPACE ERROR] Not enough free space to copy {model_name}!\n"
                f"   Required: {nbytes / (1024*1024):.1f} MB, Available: {available / (1024*1024):.1f} MB"
                + (f" ({pending / (1024*1024):.1f} MB still needed by running copies)" if pending else "")
            )
        _space_claims[f"{local_path}.tmp"] = nbytes


def _release_space(local_path):
    with _space_lock:
        _space_claims.pop(f"{local_path}.tmp", None)


def _copy_and_link(m):
    """Copy (if needed) and link one model. Runs in a worker thread; raises on failure."""
    model_name = m["model_name"]
    src_found = m["src_found"]
    src_size = m["src_size"]
    local_path = m["local_path"]
    dest_path = m["dest_path"]
    size_mb = src_size / (1024 * 1024)

    if not m["needs_copy"]:
        _log(f"   ⚡ [SKIPPED] {model_name} ({size_mb:.1f} MB) already exists on local disk -> {local_path}")
    else:
        _claim_space(model_name, local_path, src_size)
        try:
            _log(f"   📋 [COPYING] {model_name} ({size_mb:.1f} MB) from {src_found} to {local_path}...")
            try:
                mb, elapsed, speed = copy_file_chunked(src_found, local_path)
            except Exception as e:
                raise CopyError(f"   ❌ Failed to copy {model_name}: {e}") from e
            _log(f"   ✅ [COPIED] {model_name} ({mb:.1f} MB in {elapsed:.2f}s, {speed:.1f} MB/s) -> {local_path}")
        finally:
            _release_space(local_path)

    # Point /ComfyUI/models/<type>/<file> at the local copies
    try:
        _link(local_path, dest_path)
    except Exception as e:
        raise CopyError(f"   ❌ Failed to symlink {model_name}: {e}") from e
    _log(f"   ✅ [LINKED] {model_name} ({size_mb:.1f} MB) -> {dest_path}")


def ensure_model(model_name, subfolder=None):
    """Copy one model to local disk and link it into ComfyUI if needed (used for LAZY_MODELS).

    Safe to call repeatedly: returns quickly when the link is already valid. Raises on failure.
    """
    if subfolder is None:
        subfolder = LAZY_MODELS.get(model_name) or MODEL_DESTINATIONS[model_name]
    with _ensure_lock:
        start = time.time()
        dest_path, local_path = model_paths(model_name, subfolder)
        src_found = find_source(model_name, subfolder)
        if src_found is None:
            raise FileNotFoundError(f"{model_name} not found in any of {candidate_paths(model_name, subfolder)}")
        src_size = os.path.getsize(src_found)
        info = {
            "model": model_name,
            "source": src_found,
            "local_path": local_path,
            "dest_path": dest_path,
            "size_mb": src_size / (1024 * 1024),
            "copied": False,
        }
        if not link_is_valid(dest_path, local_path, src_size):
            if not local_copy_ok(local_path, src_size):
                _claim_space(model_name, local_path, src_size)
                try:
                    _, elapsed, speed = copy_file_chunked(src_found, local_path)
                finally:
                    _release_space(local_path)
                info.update(copied=True, copy_seconds=elapsed, speed_mb_s=speed)
            _link(local_path, dest_path)
            if not link_is_valid(dest_path, local_path, src_size):
                raise IOError(f"{dest_path} does not resolve to {local_path} with {src_size} bytes after linking")
        info["seconds"] = time.time() - start
        return info


def scan_and_link():
    print("=" * 60)
    print("🔍 [SETUP_MODELS] Starting model discovery and copying...")
    print("=" * 60)

    try:
        print("\n💾 System Mounts & Disk Space:")
        print(subprocess.getoutput("df -h"))
        print("\n🔍 Mount points:")
        print(subprocess.getoutput("grep -E 'volume|workspace|s3|fuse|nfs' /proc/mounts || head -n 10 /proc/mounts"))
    except Exception as me:
        print(f"Error checking mounts: {me}")

    # 1. Direct standard path checks and copying to local container disk
    print("\n🎯 Verifying / Copying required models:")
    missing_models = []
    models_to_process = []
    lazy_found = []
    total_bytes_needed = 0
    lazy_bytes_needed = 0

    for model_name, subfolder in MODEL_DESTINATIONS.items():
        dest_path, local_path = model_paths(model_name, subfolder)

        # Candidate paths in volume/workspace
        src_found = find_source(model_name, subfolder)

        if src_found:
            src_size = os.path.getsize(src_found)
            # Skip files whose local copy already exists with the same size
            local_exists = local_copy_ok(local_path, src_size)
            needs_copy = not local_exists
            if model_name in LAZY_MODELS and needs_copy:
                # Copied on the first job that needs it (an existing local copy is linked right away)
                lazy_bytes_needed += src_size
                lazy_found.append((model_name, src_found, src_size))
                continue
            if needs_copy:
                total_bytes_needed += src_size
            models_to_process.append({
                "model_name": model_name,
                "subfolder": subfolder,
                "src_found": src_found,
                "src_size": src_size,
                "local_path": local_path,
                "dest_path": dest_path,
                "needs_copy": needs_copy,
            })
        else:
            missing_models.append((model_name, subfolder, dest_path, local_path))

    for model_name, src_found, src_size in lazy_found:
        print(f"   💤 [LAZY] {model_name} ({src_size / (1024 * 1024):.1f} MB) found at {src_found}; "
              f"copied on the first job that needs it")

    # Check free disk space before copying
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    free_bytes = shutil.disk_usage(LOCAL_MODELS_DIR).free
    free_gb = free_bytes / (1024 ** 3)
    needed_gb = total_bytes_needed / (1024 ** 3)
    print(f"\n💾 Disk space check for local copy ({LOCAL_MODELS_DIR}):")
    print(f"   Available: {free_gb:.2f} GB | Required to copy: {needed_gb:.2f} GB")
    if lazy_bytes_needed:
        print(f"   Lazy models (copied later): {lazy_bytes_needed / (1024 ** 3):.2f} GB")

    if free_bytes < total_bytes_needed:
        error_msg = (
            f"❌ [DISK SPACE ERROR] Insufficient disk space on {LOCAL_MODELS_DIR}!\n"
            f"   Required: {needed_gb:.2f} GB\n"
            f"   Available: {free_gb:.2f} GB\n"
            f"   Shortage: {(total_bytes_needed - free_bytes) / (1024**3):.2f} GB\n"
            f"   Cannot copy required models to container local storage."
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    if free_bytes < total_bytes_needed + lazy_bytes_needed:
        print(
            f"   ⚠️ [DISK SPACE WARNING] Startup models fit, but lazy models need "
            f"{lazy_bytes_needed / (1024 ** 3):.2f} GB more; the first multi job will fail with a disk space error.",
            file=sys.stderr,
        )

    # Perform the copies in parallel (largest first) and point ComfyUI models to local copies
    ordered = sorted(models_to_process, key=lambda m: m["src_size"] if m["needs_copy"] else 0, reverse=True)
    n_copy = sum(1 for m in ordered if m["needs_copy"])
    print(f"\n🚀 Copying {n_copy} file(s), {COPY_FILE_WORKERS} at a time "
          f"(COPY_THREADS={COPY_THREADS} ranges per file, COPY_MAX_INFLIGHT={COPY_MAX_INFLIGHT} total):")
    copy_start = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=COPY_FILE_WORKERS) as pool:
        futures = [pool.submit(_copy_and_link, m) for m in ordered]
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            try:
                fut.result()
            except Exception as e:
                failures.append(e)
                if not _abort_copies.is_set():
                    # Stop the other copies early; the worker exits anyway
                    _abort_copies.set()
                    for other in futures:
                        other.cancel()
    if failures:
        # The first failure is the cause; the others are usually "Copy aborted"
        _log(str(failures[0]), err=True)
        if len(failures) > 1:
            _log(f"   ({len(failures) - 1} other copy/copies stopped)", err=True)
        sys.exit(1)
    copy_elapsed = time.time() - copy_start
    copied_mb = total_bytes_needed / (1024 * 1024)
    print(f"   ⏱️ [COPY] {copied_mb:.1f} MB in {copy_elapsed:.2f}s "
          f"({(copied_mb / copy_elapsed) if copy_elapsed > 0 else 0.0:.1f} MB/s aggregate)")

    # Final verification: every link resolves to its local copy with the source size
    for m in models_to_process:
        if not link_is_valid(m["dest_path"], m["local_path"], m["src_size"]):
            print(f"   ❌ [VERIFY FAILED] {m['dest_path']} does not resolve to {m['local_path']} "
                  f"with {m['src_size']} bytes", file=sys.stderr)
            sys.exit(1)
    print(f"   ✅ [VERIFIED] {len(models_to_process)} model link(s) point to local copies with matching sizes")

    # 2. Handle missing models (auto-download helper audio models if needed)
    if missing_models:
        print("\n⚠️ Handling missing models:")
        auto_download_urls = {
            "MelBandRoformer_fp16.safetensors": "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors",
            "wav2vec2-chinese-base_fp16.safetensors": "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
        }

        for item in missing_models:
            model_name, subfolder, dest_path, local_path = item
            if model_name in auto_download_urls:
                url = auto_download_urls[model_name]
                print(f"   ⬇️ [AUTO-DOWNLOAD] Downloading {model_name}...")
                vol_target = os.path.join(VOLUME_ROOT, "models", subfolder, model_name)
                save_path = vol_target if os.path.exists(os.path.join(VOLUME_ROOT, "models")) else local_path
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    import urllib.request
                    urllib.request.urlretrieve(url, save_path)
                    print(f"   ✅ Downloaded {model_name} to {save_path}")
                    if save_path != local_path:
                        mb, elapsed, speed = copy_file_chunked(save_path, local_path)
                        print(f"   ✅ [COPIED] {model_name} ({mb:.1f} MB in {elapsed:.2f}s, {speed:.1f} MB/s) -> {local_path}")
                    _link(local_path, dest_path)
                    print(f"   ✅ [LINKED] {model_name} -> {dest_path}")
                except Exception as de:
                    print(f"   ❌ Failed auto-download for {model_name}: {de}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"   ❌ [CRITICAL MISSING] {model_name} (expected in {dest_path})", file=sys.stderr)
                sys.exit(1)

    # 3. Summary of ComfyUI model directories
    print("\n📊 Final ComfyUI models directory status:")
    if os.path.exists(COMFY_MODELS_DIR):
        for sub in sorted(os.listdir(COMFY_MODELS_DIR)):
            sub_path = os.path.join(COMFY_MODELS_DIR, sub)
            if os.path.isdir(sub_path):
                models_in_sub = [f for f in os.listdir(sub_path) if not f.startswith("put_")]
                if models_in_sub:
                    print(f"   📁 {sub}: {models_in_sub}")
    print("=" * 60)

if __name__ == "__main__":
    scan_and_link()
