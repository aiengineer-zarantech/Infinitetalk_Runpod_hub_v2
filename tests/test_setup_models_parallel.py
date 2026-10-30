# ==============================================================================
# setup_models.py: parallel startup copy, lazy Multi model, verify pass, ensure_model().
# Runs the real copy code against temp folders. On Windows (no os.pread/pwrite, no symlink
# privilege) a POSIX shim is installed into setup_models' own `os` reference only.
# ==============================================================================
import os
import threading
import time
import types

import pytest

import setup_models

MULTI = "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"
STARTUP_MODELS = [m for m in setup_models.MODEL_DESTINATIONS if m != MULTI]


def _symlinks_work(tmp_path):
    target, link = tmp_path / "t", tmp_path / "l"
    target.write_bytes(b"x")
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


def _posix_shim():
    """os replacement for setup_models: pread/pwrite via a locked seek, symlinks via a registry."""
    links = {}
    io_lock = threading.Lock()
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731

    fake_path = types.ModuleType("fake_os_path")
    fake_path.__dict__.update(os.path.__dict__)
    fake_path.islink = lambda p: norm(p) in links
    fake_path.exists = lambda p: norm(p) in links or os.path.exists(p)
    fake_path.realpath = lambda p: os.path.realpath(links.get(norm(p), p))

    fake_os = types.ModuleType("fake_os")
    fake_os.__dict__.update(os.__dict__)
    fake_os.path = fake_path

    def symlink(src, dst):
        if fake_path.exists(dst):
            raise FileExistsError(dst)
        links[norm(dst)] = src

    def remove(p):
        if links.pop(norm(p), None) is None:
            os.remove(p)

    def pread(fd, n, offset):
        with io_lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)

    def pwrite(fd, data, offset):
        with io_lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.write(fd, data)

    fake_os.symlink = symlink
    fake_os.remove = remove
    fake_os.pread = pread
    fake_os.pwrite = pwrite
    fake_os.open = lambda p, flags, mode=0o777: os.open(p, flags | getattr(os, "O_BINARY", 0), mode)
    return fake_os


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp volume with all 9 models (distinct sizes) and empty local/ComfyUI folders."""
    if not (hasattr(os, "pread") and _symlinks_work(tmp_path)):
        monkeypatch.setattr(setup_models, "os", _posix_shim())
    volume = tmp_path / "runpod-volume"
    sizes = {}
    for i, (name, sub) in enumerate(setup_models.MODEL_DESTINATIONS.items()):
        path = volume / "models" / sub / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sizes[name] = 3000 + 997 * i
        path.write_bytes(bytes((j * 7 + i) % 251 for j in range(sizes[name])))
    monkeypatch.setattr(setup_models, "SOURCE_ROOTS", [str(volume), str(tmp_path / "workspace")])
    monkeypatch.setattr(setup_models, "VOLUME_ROOT", str(volume))
    monkeypatch.setattr(setup_models, "COMFY_MODELS_DIR", str(tmp_path / "ComfyUI" / "models"))
    monkeypatch.setattr(setup_models, "LOCAL_MODELS_DIR", str(tmp_path / "models_local"))
    monkeypatch.setattr(setup_models, "COPY_PART_SIZE", 1024)  # several ranges per file
    monkeypatch.setattr(setup_models, "_space_claims", {})
    setup_models._abort_copies.clear()
    yield types.SimpleNamespace(tmp=tmp_path, volume=volume, sizes=sizes)
    setup_models._abort_copies.clear()


def paths(name):
    sub = setup_models.MODEL_DESTINATIONS[name]
    return (os.path.join(setup_models.COMFY_MODELS_DIR, sub, name),
            os.path.join(setup_models.LOCAL_MODELS_DIR, sub, name))


def track_copies(monkeypatch, delay=0.15, fail_on=None):
    state = {"active": 0, "max": 0, "copied": []}
    lock = threading.Lock()
    real = setup_models.copy_file_chunked

    def tracked(src, dst):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        try:
            time.sleep(delay)
            if fail_on and os.path.basename(dst) == fail_on:
                raise IOError("simulated GeeseFS read error")
            result = real(src, dst)
            with lock:
                state["copied"].append(os.path.basename(dst))
            return result
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(setup_models, "copy_file_chunked", tracked)
    return state


def test_clamp_file_workers():
    assert setup_models._clamp_file_workers("2") == 4
    assert setup_models._clamp_file_workers("6") == 6
    assert setup_models._clamp_file_workers("20") == 8
    assert setup_models._clamp_file_workers("abc") == 6
    assert 4 <= setup_models.COPY_FILE_WORKERS <= 8


def test_candidate_path_order_unchanged():
    assert setup_models.candidate_paths("m.safetensors", "vae") == [
        os.path.join("/runpod-volume", "models", "vae", "m.safetensors"),
        os.path.join("/runpod-volume", "vae", "m.safetensors"),
        os.path.join("/runpod-volume", "m.safetensors"),
        os.path.join("/workspace", "models", "vae", "m.safetensors"),
        os.path.join("/workspace", "vae", "m.safetensors"),
        os.path.join("/workspace", "m.safetensors"),
    ]


def test_startup_copies_in_parallel_and_skips_multi(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models, "COPY_FILE_WORKERS", 6)
    state = track_copies(monkeypatch)
    setup_models.scan_and_link()
    out = capsys.readouterr().out

    assert 4 <= state["max"] <= 8
    assert state["max"] == 6
    assert sorted(state["copied"]) == sorted(STARTUP_MODELS)
    for name in STARTUP_MODELS:
        dest, local = paths(name)
        assert setup_models.link_is_valid(dest, local, env.sizes[name]), name
        assert open(local, "rb").read() == (env.volume / "models" / setup_models.MODEL_DESTINATIONS[name] / name).read_bytes()
    dest, local = paths(MULTI)
    assert not setup_models.os.path.islink(dest)
    assert not os.path.exists(local)
    assert f"[LAZY] {MULTI}" in out
    assert "[VERIFIED] 8 model link(s)" in out


def test_largest_files_start_first(env, monkeypatch):
    monkeypatch.setattr(setup_models, "COPY_FILE_WORKERS", 4)
    order = []
    real = setup_models.copy_file_chunked
    monkeypatch.setattr(setup_models, "copy_file_chunked",
                        lambda s, d: (order.append(os.path.basename(d)), real(s, d))[1])
    setup_models.scan_and_link()
    expected = sorted(STARTUP_MODELS, key=lambda n: env.sizes[n], reverse=True)
    # The first 4 submitted are the 4 largest (thread start order may interleave within that group)
    assert set(order[:4]) == set(expected[:4])


def test_missing_multi_on_volume_is_still_critical(env, capsys):
    (env.volume / "models" / "diffusion_models" / MULTI).unlink()
    with pytest.raises(SystemExit) as exc:
        setup_models.scan_and_link()
    assert exc.value.code == 1
    assert f"[CRITICAL MISSING] {MULTI}" in capsys.readouterr().err


def test_existing_local_copy_is_skipped_and_linked(env, monkeypatch, capsys):
    name = STARTUP_MODELS[0]
    dest, local = paths(name)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        f.write(b"\0" * env.sizes[name])
    state = track_copies(monkeypatch, delay=0)
    setup_models.scan_and_link()
    assert name not in state["copied"]
    assert "[SKIPPED] " + name in capsys.readouterr().out
    assert setup_models.link_is_valid(dest, local, env.sizes[name])


def test_existing_local_multi_is_linked_at_startup(env, capsys):
    dest, local = paths(MULTI)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        f.write(b"\0" * env.sizes[MULTI])
    setup_models.scan_and_link()
    assert setup_models.link_is_valid(dest, local, env.sizes[MULTI])
    assert "[LAZY]" not in capsys.readouterr().out


def test_copy_failure_exits_and_stops_other_copies(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models, "COPY_FILE_WORKERS", 4)
    largest = max(STARTUP_MODELS, key=lambda n: env.sizes[n])
    state = track_copies(monkeypatch, delay=0.2, fail_on=largest)
    with pytest.raises(SystemExit) as exc:
        setup_models.scan_and_link()
    assert exc.value.code == 1
    assert f"Failed to copy {largest}: simulated GeeseFS read error" in capsys.readouterr().err
    # Copies queued behind the failure were cancelled, so not every file was copied
    assert len(state["copied"]) < len(STARTUP_MODELS) - 1


def test_size_mismatch_after_copy_fails(env, monkeypatch, capsys):
    real_getsize = setup_models.os.path.getsize

    def short_tmp(p):
        return real_getsize(p) - 1 if str(p).endswith(".tmp") else real_getsize(p)

    monkeypatch.setattr(setup_models.os.path, "getsize", short_tmp)
    with pytest.raises(SystemExit):
        setup_models.scan_and_link()
    assert "Size mismatch after copy" in capsys.readouterr().err


def test_verify_pass_rejects_bad_link(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models, "link_is_valid", lambda d, l, s: False)
    with pytest.raises(SystemExit):
        setup_models.scan_and_link()
    assert "[VERIFY FAILED]" in capsys.readouterr().err


def test_upfront_disk_check_fails(env, monkeypatch, capsys):
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=1000))
    with pytest.raises(SystemExit):
        setup_models.scan_and_link()
    assert "[DISK SPACE ERROR] Insufficient disk space" in capsys.readouterr().err


def test_lazy_space_warning(env, monkeypatch, capsys):
    startup = sum(env.sizes[n] for n in STARTUP_MODELS)
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=startup + 10))
    monkeypatch.setattr(setup_models, "_claim_space", lambda *a: None)
    setup_models.scan_and_link()
    assert "[DISK SPACE WARNING]" in capsys.readouterr().err


def test_claim_space_counts_running_copies(env, monkeypatch):
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=150))
    monkeypatch.setattr(setup_models, "_allocated_bytes", lambda p: 0)
    setup_models._claim_space("a", "/x/a", 100)
    with pytest.raises(setup_models.DiskSpaceError, match="still needed by running copies"):
        setup_models._claim_space("b", "/x/b", 100)
    setup_models._release_space("/x/a")
    setup_models._claim_space("b", "/x/b", 100)


def test_ensure_model_copies_links_and_is_idempotent(env):
    dest, local = paths(MULTI)
    info = setup_models.ensure_model(MULTI)
    assert info["copied"] is True
    assert info["dest_path"] == dest and info["local_path"] == local
    assert setup_models.link_is_valid(dest, local, env.sizes[MULTI])
    again = setup_models.ensure_model(MULTI)
    assert again["copied"] is False


def test_ensure_model_missing_source(env):
    (env.volume / "models" / "diffusion_models" / MULTI).unlink()
    with pytest.raises(FileNotFoundError):
        setup_models.ensure_model(MULTI)


def test_ensure_model_disk_full(env, monkeypatch):
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10))
    with pytest.raises(setup_models.DiskSpaceError):
        setup_models.ensure_model(MULTI)
