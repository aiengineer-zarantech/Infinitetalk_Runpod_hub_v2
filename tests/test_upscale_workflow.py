# ==============================================================================
# v26: every SeedVR2 segment prompt the upscale stage builds (first try and the 3 OOM retries, both models,
# both orientations) uses only nodes, inputs, combo values and ranges that exist in the /object_info of the
# installed versions (tests/fixtures/object_info_upscale.json, taken from image v25). The Dockerfile.v26
# build repeats this against a live ComfyUI (scripts/smoke_v26.py).
# ==============================================================================
import json
from pathlib import Path

import pytest

from worker_guard import upscale

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJECT_INFO = json.loads((REPO_ROOT / "tests" / "fixtures" / "object_info_upscale.json").read_text(encoding="utf-8"))


def _spec(info, name):
    for section in ("required", "optional"):
        if name in (info["input"].get(section) or {}):
            return info["input"][section][name]
    return None


def _dynamic_spec(info, name):
    """'chunking_mode.frames_per_chunk' -> the spec inside the selected DynamicCombo option."""
    base, _, sub = name.partition(".")
    spec = _spec(info, base)
    if not spec or spec[0] != "COMFY_DYNAMICCOMBO_V3":
        return None
    for option in spec[1]["options"]:
        for section in option["inputs"].values():
            if sub in section:
                return section[sub]
    return None


def _vhs_format_spec(info, fmt, name):
    widgets = _spec(info, "format")[1]["formats"].get(fmt) or []
    for widget in widgets:
        if widget[0] == name:
            return widget[1:]
    return None


def check_value(node_id, name, spec, value):
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, list):
        if kind:  # empty: model lists (no model files in the fixture's ComfyUI)
            assert value in kind, f"node {node_id} {name}={value!r} not in {kind}"
    elif kind == "COMBO":
        options = opts.get("options") or []
        if options:
            assert value in options, f"node {node_id} {name}={value!r} not in {options}"
    elif kind in ("INT", "FLOAT"):
        assert isinstance(value, (int, float)) and not isinstance(value, bool), f"node {node_id} {name}={value!r}"
        if "min" in opts:
            assert value >= opts["min"], f"node {node_id} {name}={value} < min {opts['min']}"
        if "max" in opts:
            assert value <= opts["max"], f"node {node_id} {name}={value} > max {opts['max']}"
    elif kind == "BOOLEAN":
        assert isinstance(value, bool), f"node {node_id} {name}={value!r}"
    elif kind == "STRING":
        assert isinstance(value, str), f"node {node_id} {name}={value!r}"
    elif kind == "COMFY_DYNAMICCOMBO_V3":
        assert value in [o["key"] for o in opts["options"]], f"node {node_id} {name}={value!r}"


def validate(prompt):
    for node_id, node in prompt.items():
        info = OBJECT_INFO.get(node["class_type"])
        assert info is not None, f"node {node_id}: {node['class_type']} not in object_info"
        given = node["inputs"]
        for section in ("required",):
            for name in info["input"].get(section) or {}:
                assert name in given, f"node {node_id} ({node['class_type']}) misses required input {name}"
        for name, value in given.items():
            spec = _spec(info, name)
            if spec is None and "." in name:
                spec = _dynamic_spec(info, name)
            if spec is None and node["class_type"] == "VHS_VideoCombine":
                spec = _vhs_format_spec(info, given["format"], name)
                if spec is None and name in ("save_metadata", "trim_to_audio"):
                    spec = ["BOOLEAN"]  # VHS adds these for every video format
            assert spec is not None, f"node {node_id} ({node['class_type']}): unknown input {name}"
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                source = prompt.get(value[0])
                assert source is not None, f"node {node_id} {name} links to missing node {value[0]}"
                outputs = OBJECT_INFO[source["class_type"]]["output"]
                assert value[1] < len(outputs), f"node {node_id} {name} links to output {value[1]} of {outputs}"
                continue
            if upscale.LOADERS.get(node["class_type"]) == name:
                continue  # model file lists: the fixture's ComfyUI had no model files (smoke_v26.py checks them live)
            check_value(node_id, name, spec, value)


SEGMENT = upscale.Segment(index=1, start=41, end=82, load_start=33, load_end=90)


@pytest.mark.parametrize("attempt", (upscale.FIRST_ATTEMPT,) + upscale.OOM_RETRIES)
@pytest.mark.parametrize("model", sorted(upscale.MODEL_FILES.values()))
@pytest.mark.parametrize("target", [upscale.LANDSCAPE, upscale.PORTRAIT])
def test_segment_prompts_match_object_info(attempt, model, target):
    template = upscale.load_template(str(REPO_ROOT))
    prompt = upscale.build_prompt(template, "/ComfyUI/temp/in.mp4", SEGMENT, target, model, attempt, 25.0, "lab", "p")
    validate(prompt)
    assert prompt["1"]["inputs"]["skip_first_frames"] == 33 and prompt["1"]["inputs"]["frame_load_cap"] == 57
    assert (prompt["2"]["inputs"]["width"], prompt["2"]["inputs"]["height"]) == target
    assert prompt["2"]["inputs"]["crop"] == "center"
    assert prompt["7"]["inputs"]["unet_name"] == model
    chunk = prompt["6"]["inputs"]
    if attempt.chunk:
        assert chunk["chunking_mode"] == "manual" and chunk["chunking_mode.frames_per_chunk"] == attempt.chunk
    else:
        assert chunk["chunking_mode"] == "auto" and "chunking_mode.frames_per_chunk" not in chunk
    for node in ("5", "11"):
        assert prompt[node]["inputs"]["tile_size"] == attempt.tile
        assert prompt[node]["inputs"]["temporal_size"] == attempt.temporal
    assert template == upscale.load_template(str(REPO_ROOT))  # the template itself is never changed


def test_oom_retries_get_smaller():
    chunks = [a.chunk for a in upscale.OOM_RETRIES]
    tiles = [a.tile for a in upscale.OOM_RETRIES]
    assert chunks == sorted(chunks, reverse=True) and all((c - 1) % 4 == 0 for c in chunks)
    assert tiles == sorted(tiles, reverse=True)
    assert all(a.overlap < a.tile and a.temporal_overlap < a.temporal for a in (upscale.FIRST_ATTEMPT,) + upscale.OOM_RETRIES)


def test_template_follows_the_comfyui_blueprint_graph():
    """Same wiring as ComfyUI's 'Upscale Video (SeedVR2 3B Int8)' blueprint (one step, euler, cfg 1, fixed seed)."""
    p = upscale.load_template(str(REPO_ROOT))
    assert p["3"]["inputs"]["resized_images"] == ["2", 0]
    assert p["5"]["inputs"]["pixels"] == ["3", 0]
    assert p["6"]["inputs"]["latent"] == ["5", 0]
    assert p["8"]["inputs"]["vae_conditioning"] == ["6", 0] and p["9"]["inputs"]["latent_image"] == ["6", 0]
    assert p["9"]["inputs"]["positive"] == ["8", 0] and p["9"]["inputs"]["negative"] == ["8", 1]
    assert {k: p["9"]["inputs"][k] for k in ("steps", "cfg", "sampler_name", "scheduler", "denoise")} == {
        "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}
    assert p["10"]["inputs"] == {"latents": ["9", 0], "temporal_overlap": ["6", 1]}
    assert p["11"]["inputs"]["samples"] == ["10", 0]
    assert p["12"]["inputs"]["images"] == ["11", 0] and p["12"]["inputs"]["original_resized_images"] == ["2", 0]
    assert p["13"]["inputs"]["format"] == "video/ffv1-mkv" and "audio" not in p["13"]["inputs"]
