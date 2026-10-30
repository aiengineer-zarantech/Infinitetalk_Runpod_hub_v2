# ==============================================================================
# v26 setup_models: SeedVR2 models come from <volume>/models/upscale/ and are linked into ComfyUI
# diffusion_models/ and vae/. 3B + VAE at startup (in parallel with the other copies), 7B on first use.
# Missing on the volume: downloaded once from the pinned Hugging Face revision, sha256 checked before the
# final name exists. A failed upscale model never stops the worker.
# ==============================================================================
import hashlib
import io
import os
import urllib.request

import pytest

import setup_models
from test_setup_models_parallel import env  # noqa: F401  (temp volume, local disk and ComfyUI folders)

CONTENT = {
    "seedvr2_3b_fp16.safetensors": b"3b" * 3000,
    "seedvr2_ema_vae_fp16.safetensors": b"vae" * 700,
    "seedvr2_7b_fp16.safetensors": b"7b" * 5000,
}


@pytest.fixture
def small_models(monkeypatch):
    specs = {}
    for name, spec in setup_models.UPSCALE_MODELS.items():
        data = CONTENT[name]
        specs[name] = {**spec, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(setup_models, "UPSCALE_MODELS", specs)
    monkeypatch.setenv("UPSCALE_MODELS_AT_STARTUP", "1")
    return specs


@pytest.fixture
def downloads(monkeypatch):
    """Fake Hugging Face: serves CONTENT (or `served` overrides); records the URLs."""
    state = {"urls": [], "served": {}}

    def urlopen(url, timeout=None):
        state["urls"].append(url)
        name = url.rsplit("/", 1)[1]
        return io.BytesIO(state["served"].get(name, CONTENT[name]))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return state


def put_on_volume(env, name):
    path = env.volume / "models" / "upscale" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CONTENT[name])
    return path


def comfy_path(name):
    folder = setup_models.UPSCALE_MODELS[name]["folder"]
    return os.path.join(setup_models.COMFY_MODELS_DIR, folder, name)


def linked(name):
    path = comfy_path(name)
    return setup_models.os.path.islink(path) and open(setup_models.os.path.realpath(path), "rb").read() == CONTENT[name]


def test_download_url_is_pinned():
    url = setup_models.upscale_download_url("seedvr2_3b_fp16.safetensors")
    assert url == ("https://huggingface.co/Comfy-Org/SeedVR2/resolve/df48879708206a403d2a61acd55578c2e80fd233/"
                   "diffusion_models/seedvr2_3b_fp16.safetensors")
    assert set(setup_models.UPSCALE_MODELS) == {"seedvr2_3b_fp16.safetensors", "seedvr2_7b_fp16.safetensors",
                                                "seedvr2_ema_vae_fp16.safetensors"}
    assert [n for n, s in setup_models.UPSCALE_MODELS.items() if s["startup"]] == [
        "seedvr2_3b_fp16.safetensors", "seedvr2_ema_vae_fp16.safetensors"]


def test_startup_links_3b_and_vae_from_the_volume(env, small_models, downloads, capsys):
    for name in CONTENT:
        put_on_volume(env, name)
    setup_models.scan_and_link()
    out = capsys.readouterr().out
    assert linked("seedvr2_3b_fp16.safetensors") and linked("seedvr2_ema_vae_fp16.safetensors")
    assert not setup_models.os.path.exists(comfy_path("seedvr2_7b_fp16.safetensors"))  # lazy
    assert "[LAZY] seedvr2_7b_fp16.safetensors" in out
    assert downloads["urls"] == []
    for name in setup_models.MODEL_DESTINATIONS:  # the InfiniteTalk startup models are unaffected
        if name not in setup_models.LAZY_MODELS:
            assert setup_models.os.path.islink(os.path.join(setup_models.COMFY_MODELS_DIR,
                                                            setup_models.MODEL_DESTINATIONS[name], name))


def test_missing_models_are_downloaded_once_to_the_volume(env, small_models, downloads, capsys):
    setup_models.scan_and_link()
    assert sorted(u.rsplit("/", 1)[1] for u in downloads["urls"]) == [
        "seedvr2_3b_fp16.safetensors", "seedvr2_ema_vae_fp16.safetensors"]
    for name in ("seedvr2_3b_fp16.safetensors", "seedvr2_ema_vae_fp16.safetensors"):
        assert (env.volume / "models" / "upscale" / name).read_bytes() == CONTENT[name]
        assert linked(name)
    setup_models.ensure_upscale_model("seedvr2_3b_fp16.safetensors")  # already there: no second download
    assert len(downloads["urls"]) == 2


def test_7b_on_first_use(env, small_models, downloads):
    info = setup_models.ensure_upscale_model("seedvr2_7b_fp16.safetensors")
    assert info["copied"] and linked("seedvr2_7b_fp16.safetensors")
    assert [u.rsplit("/", 1)[1] for u in downloads["urls"]] == ["seedvr2_7b_fp16.safetensors"]


def test_bad_download_never_gets_the_final_name_and_does_not_stop_startup(env, small_models, downloads, capsys):
    downloads["served"]["seedvr2_3b_fp16.safetensors"] = b"3b" * 2999 + b"XX"  # same size, wrong sha256
    setup_models.scan_and_link()  # no SystemExit
    captured = capsys.readouterr()
    folder = env.volume / "models" / "upscale"
    assert not (folder / "seedvr2_3b_fp16.safetensors").exists()
    assert not (folder / "seedvr2_3b_fp16.safetensors.tmp").exists()
    assert "[UPSCALE] seedvr2_3b_fp16.safetensors not prepared: IOError" in captured.err or \
           "[UPSCALE] seedvr2_3b_fp16.safetensors not prepared: OSError" in captured.err
    assert "sha256" in captured.err
    assert linked("seedvr2_ema_vae_fp16.safetensors")


def test_wrong_size_on_the_volume_is_rejected(env, small_models, downloads):
    path = put_on_volume(env, "seedvr2_ema_vae_fp16.safetensors")
    path.write_bytes(b"short")
    with pytest.raises(IOError, match="expected"):
        setup_models.ensure_upscale_model("seedvr2_ema_vae_fp16.safetensors")


def test_startup_switch_off(env, small_models, downloads, monkeypatch, capsys):
    monkeypatch.setenv("UPSCALE_MODELS_AT_STARTUP", "0")
    setup_models.scan_and_link()
    assert downloads["urls"] == [] and not setup_models.os.path.exists(comfy_path("seedvr2_3b_fp16.safetensors"))
    assert "UPSCALE_MODELS_AT_STARTUP=0" in capsys.readouterr().out
