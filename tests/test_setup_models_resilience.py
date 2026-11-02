# ==============================================================================
# setup_models.py (perf/5090-speedup): range retry, resumable file copy, file retry, disk check with the
# startup upscale bytes, and MODEL_SOURCE=hf_cache (RunPod Cached Models, no copy).
# Runs the real copy code against temp folders (fixture `env` from test_setup_models_parallel).
# ==============================================================================
import errno
import os
import types

import pytest

import setup_models
from test_setup_models_parallel import MULTI, STARTUP_MODELS, env, paths, track_copies  # noqa: F401

REPO = "zarantech/infinitetalk-models"
HASH = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("MODEL_SOURCE", "HF_CACHE_REPO", "HF_CACHE_ROOT", "HF_CACHE_REVISION", "HF_CACHE_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(setup_models, "COPY_RETRY_BASE_SEC", 0.0)
    for key in setup_models._stats:
        setup_models._stats[key] = 0


def flaky_pread(monkeypatch, should_fail):
    """os.pread that raises EIO while should_fail(offset, call_number) is true."""
    real = setup_models.os.pread
    calls = {"n": 0, "failed": 0}

    def pread(fd, n, offset):
        calls["n"] += 1
        if should_fail(offset, calls["n"]):
            calls["failed"] += 1
            raise OSError(errno.EIO, "Input/output error (simulated GeeseFS)")
        return real(fd, n, offset)

    monkeypatch.setattr(setup_models.os, "pread", pread)
    return calls


def same_bytes(env, name):
    _, local = paths(name)
    sub = setup_models.MODEL_DESTINATIONS[name]
    with open(local, "rb") as a, open(env.volume / "models" / sub / name, "rb") as b:
        return a.read() == b.read()


# ---------------------------------------------------------------- volume: retry and resume
def test_transient_read_errors_are_retried(env, monkeypatch, capsys):
    calls = flaky_pread(monkeypatch, lambda off, n: n in (3, 7, 8))
    setup_models.scan_and_link()
    out = capsys.readouterr()
    assert calls["failed"] == 3 and setup_models._stats["range_retries"] == 3
    assert out.err.count("[RETRY] read at offset") == 3
    assert all(same_bytes(env, n) for n in STARTUP_MODELS)
    assert "range retries 3, file retries 0" in out.out and "[TIMING] model setup" in out.out


def test_read_that_keeps_failing_fails_after_all_retries(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models, "COPY_RANGE_RETRIES", 2)
    monkeypatch.setattr(setup_models, "COPY_FILE_RETRIES", 2)
    largest = max(STARTUP_MODELS, key=lambda n: env.sizes[n])
    big_fd = {}
    real_open = setup_models.os.open

    def tracking_open(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        if os.path.basename(str(path)) == largest and not flags & os.O_WRONLY:
            big_fd[fd] = True
        return fd
    monkeypatch.setattr(setup_models.os, "open", tracking_open)
    real_pread = setup_models.os.pread

    def pread(fd, n, offset):
        if fd in big_fd and offset >= 2048:
            raise OSError(errno.EIO, "Input/output error (simulated GeeseFS)")
        return real_pread(fd, n, offset)
    monkeypatch.setattr(setup_models.os, "pread", pread)
    with pytest.raises(SystemExit) as exc:
        setup_models.scan_and_link()
    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert f"Failed to copy {largest} after 2 attempt(s): read at offset" in err
    assert "failed 3 time(s)" in err                                # 1 read + COPY_RANGE_RETRIES=2


def test_resume_copies_only_the_missing_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_models, "COPY_PART_SIZE", 1000)
    monkeypatch.setattr(setup_models, "COPY_THREADS", 1)          # ranges in order: deterministic progress
    monkeypatch.setattr(setup_models, "COPY_RANGE_RETRIES", 0)
    src, dst = tmp_path / "src.bin", tmp_path / "local" / "dst.bin"
    data = bytes(i % 251 for i in range(9500))                     # 10 ranges
    src.write_bytes(data)
    setup_models._abort_copies.clear()
    flaky_pread(monkeypatch, lambda off, n: off >= 5000)
    with pytest.raises(IOError):
        setup_models.copy_file_chunked(str(src), str(dst))
    progress = (tmp_path / "local" / "dst.bin.tmp.progress").read_text().splitlines()
    assert sorted(int(x) for x in progress[1:]) == [0, 1, 2, 3, 4]

    monkeypatch.undo()                                              # real pread again
    monkeypatch.setattr(setup_models, "COPY_PART_SIZE", 1000)
    monkeypatch.setattr(setup_models, "COPY_THREADS", 1)
    read_offsets = []
    real_range = setup_models._copy_range
    monkeypatch.setattr(setup_models, "_copy_range",
                        lambda s, d, off, ln, p=None: read_offsets.append(off) or real_range(s, d, off, ln, p))
    setup_models.copy_file_chunked(str(src), str(dst))
    assert sorted(read_offsets) == [5000, 6000, 7000, 8000, 9000]
    assert dst.read_bytes() == data
    assert not (tmp_path / "local" / "dst.bin.tmp").exists()
    assert not (tmp_path / "local" / "dst.bin.tmp.progress").exists()


def test_changed_source_is_not_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_models, "COPY_PART_SIZE", 1000)
    src, dst = tmp_path / "src.bin", tmp_path / "local" / "dst.bin"
    src.write_bytes(b"a" * 5000)
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "dst.bin.tmp").write_bytes(b"x" * 4000)
    (tmp_path / "local" / "dst.bin.tmp.progress").write_text('{"size": 4000, "part": 1000, "mtime": 1}\n0\n1\n')
    setup_models.copy_file_chunked(str(src), str(dst))
    assert dst.read_bytes() == b"a" * 5000
    assert setup_models._stats["resumed_bytes"] == 0


def test_failed_file_is_retried_and_resumed(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models, "COPY_RANGE_RETRIES", 0)
    monkeypatch.setattr(setup_models, "COPY_THREADS", 1)
    failures = {"left": 1}
    real = setup_models.os.pread

    def pread(fd, n, offset):
        if offset == 2048 and failures["left"]:
            failures["left"] -= 1
            raise OSError(errno.EIO, "simulated")
        return real(fd, n, offset)
    monkeypatch.setattr(setup_models.os, "pread", pread)
    setup_models.scan_and_link()
    out = capsys.readouterr()
    assert setup_models._stats["file_retries"] == 1 and setup_models._stats["resumed_bytes"] >= 2048
    assert "[RESUME]" in out.out and "attempt 1/3 failed" in out.err
    assert all(same_bytes(env, n) for n in STARTUP_MODELS)


def test_startup_upscale_copy_is_skipped_when_it_does_not_fit(env, monkeypatch, capsys):
    monkeypatch.setenv("UPSCALE_MODELS_AT_STARTUP", "1")
    started = []
    monkeypatch.setattr(setup_models, "prepare_startup_upscale_models", lambda: started.append(1))
    need = sum(env.sizes[n] for n in STARTUP_MODELS)
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=need + 10_000))
    setup_models.scan_and_link()
    assert not started
    assert "[UPSCALE] startup copy skipped" in capsys.readouterr().err
    assert all(same_bytes(env, n) for n in STARTUP_MODELS)


def test_startup_upscale_copy_runs_when_it_fits(env, monkeypatch):
    monkeypatch.setenv("UPSCALE_MODELS_AT_STARTUP", "1")
    started = []
    monkeypatch.setattr(setup_models, "prepare_startup_upscale_models", lambda: started.append(1))
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10 ** 12))
    setup_models.scan_and_link()
    assert started == [1]


# ---------------------------------------------------------------- hf_cache
def make_cache(tmp_path, models=None, refs=True, extra_snapshots=()):
    """Standard HF cache layout: snapshots/<hash>/<folder>/<file> -> ../../../blobs/<id>."""
    root = tmp_path / "hf-hub"
    base = root / ("models--" + REPO.replace("/", "--"))
    snap = base / "snapshots" / HASH
    (base / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    models = [m for m in setup_models.MODEL_DESTINATIONS if m != MULTI] if models is None else models
    for i, name in enumerate(models):
        sub = setup_models.MODEL_DESTINATIONS[name]
        blob = base / "blobs" / f"blob{i}"
        blob.write_bytes(bytes([i]) * (500 + i))
        (snap / sub).mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, snap / sub), snap / sub / name)
    for other in extra_snapshots:
        (base / "snapshots" / other).mkdir(parents=True)
    if refs:
        (base / "refs").mkdir()
        (base / "refs" / "main").write_text(HASH)
    return root, snap


def use_cache(monkeypatch, root, **extra):
    monkeypatch.setenv("MODEL_SOURCE", "hf_cache")
    monkeypatch.setenv("HF_CACHE_REPO", REPO)
    monkeypatch.setenv("HF_CACHE_ROOT", str(root))
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_hf_cache_links_every_model_and_copies_nothing(env, monkeypatch, capsys):
    root, snap = make_cache(env.tmp)
    use_cache(monkeypatch, root)
    state = track_copies(monkeypatch, delay=0)
    setup_models.scan_and_link()
    out = capsys.readouterr().out
    assert state["copied"] == []                                     # zero bytes copied
    for name in STARTUP_MODELS:
        dest, local = paths(name)
        sub = setup_models.MODEL_DESTINATIONS[name]
        assert os.path.realpath(dest) == os.path.realpath(snap / sub / name)
        assert os.path.realpath(dest).startswith(str(root)) and "/blobs/" in os.path.realpath(dest)
        assert not os.path.exists(local)
    assert f"{len(STARTUP_MODELS)} linked from the HF cache, 0.00 GB copied" in out
    assert "[LAZY]" in out and MULTI in out                         # Multi still lazy from the volume


def test_hf_cache_missing_file_uses_the_volume(env, monkeypatch, capsys):
    vae = "Wan2_1_VAE_bf16.safetensors"
    root, _ = make_cache(env.tmp, models=[m for m in STARTUP_MODELS if m != vae])
    use_cache(monkeypatch, root)
    state = track_copies(monkeypatch, delay=0)
    setup_models.scan_and_link()
    assert state["copied"] == [vae]
    assert same_bytes(env, vae)
    assert f"{vae}: not in HF cache, volume fallback" in capsys.readouterr().out


def test_hf_cache_missing_file_fails_without_fallback(env, monkeypatch, capsys):
    root, _ = make_cache(env.tmp, models=STARTUP_MODELS[1:])
    use_cache(monkeypatch, root, HF_CACHE_FALLBACK="0")
    with pytest.raises(SystemExit) as exc:
        setup_models.scan_and_link()
    assert exc.value.code == 1 and "not in the cache" in capsys.readouterr().err


def test_hf_cache_absent_falls_back_or_fails(env, monkeypatch, capsys):
    use_cache(monkeypatch, env.tmp / "no-cache-here")
    state = track_copies(monkeypatch, delay=0)
    setup_models.scan_and_link()                                    # fallback on: everything from the volume
    assert sorted(state["copied"]) == sorted(STARTUP_MODELS)
    assert "every model uses the volume" in capsys.readouterr().err
    monkeypatch.setenv("HF_CACHE_FALLBACK", "0")
    with pytest.raises(SystemExit):
        setup_models.scan_and_link()


def test_hf_cache_without_volume_multi_is_only_a_warning(env, monkeypatch, capsys):
    root, _ = make_cache(env.tmp)
    use_cache(monkeypatch, root)
    os.remove(env.volume / "models" / "diffusion_models" / MULTI)
    setup_models.scan_and_link()
    assert "is neither in the HF cache nor on a volume" in capsys.readouterr().out


def test_snapshot_resolution(tmp_path, monkeypatch):
    root, snap = make_cache(tmp_path, models=[], refs=False, extra_snapshots=("ffff",))
    use_cache(monkeypatch, root)
    with pytest.raises(setup_models.HFCacheError, match="2 snapshots"):
        setup_models.hf_snapshot_dir()
    monkeypatch.setenv("HF_CACHE_REVISION", HASH)
    assert setup_models.hf_snapshot_dir() == str(snap)
    monkeypatch.setenv("HF_CACHE_REVISION", "nope")
    with pytest.raises(setup_models.HFCacheError, match="nope not found"):
        setup_models.hf_snapshot_dir()
    monkeypatch.setenv("HF_CACHE_REPO", "no-slash")
    with pytest.raises(setup_models.HFCacheError, match="org>/<name"):
        setup_models.hf_snapshot_dir()


def test_ensure_model_prefers_the_cache(env, monkeypatch):
    root, snap = make_cache(env.tmp, models=[MULTI])
    use_cache(monkeypatch, root)
    info = setup_models.ensure_model(MULTI)
    assert info["hf_cache"] is True and info["copied"] is False
    assert os.path.realpath(paths(MULTI)[0]) == os.path.realpath(snap / "diffusion_models" / MULTI)


def test_upscale_at_startup_defaults(monkeypatch):
    monkeypatch.delenv("UPSCALE_MODELS_AT_STARTUP", raising=False)
    assert setup_models.upscale_at_startup() is True
    monkeypatch.setenv("MODEL_SOURCE", "hf_cache")
    assert setup_models.upscale_at_startup() is False
    monkeypatch.setenv("UPSCALE_MODELS_AT_STARTUP", "1")
    assert setup_models.upscale_at_startup() is True
