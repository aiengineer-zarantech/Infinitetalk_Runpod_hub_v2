# ==============================================================================
# negative_prompt input: node 241 gets the negative, a WanVideoApplyNAG node (900) is injected and the
# sampler's text_embeds is rewired to it (cfg 1.0 skips the negative pass, so NAG applies it).
# Jobs without negative_prompt are covered by test_v23_prompt_identity.py.
# ==============================================================================
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "tests" / "fixtures" / "prompt_driver.py"

IMAGE = base64.b64encode(b"\x89PNG fake image").decode()
WAV = base64.b64encode(b"RIFF fake wav").decode()
NEG = "camera shake, head shaking"
STUDIO = {"input_type": "image", "person_count": "single", "prompt": "A person talking naturally",
          "width": 720, "height": 480, "blocks_to_swap": 0, "force_offload": False, "network_volume": False,
          "image_base64": IMAGE, "wav_base64": WAV, "negative_prompt": NEG}
VIDEO = {"input_type": "video", "video_base64": base64.b64encode(b"fake mp4").decode()}
JOBS = [
    ({}, STUDIO),                                            # 0 I2V_single_compiled
    ({}, {**STUDIO, "person_count": "multi"}),               # 1 I2V_multi_compiled
    ({"TORCH_COMPILE": "0"}, STUDIO),                        # 2 I2V_single
    ({"TORCH_COMPILE": "0"}, {**STUDIO, "person_count": "multi"}),  # 3 I2V_multi
    ({}, {**STUDIO, **VIDEO}),                               # 4 V2V_single
    ({}, {**STUDIO, **VIDEO, "person_count": "multi"}),      # 5 V2V_multi
    ({}, {**STUDIO, "nag_scale": 7, "nag_tau": 2.0, "nag_alpha": 0.5}),  # 6 custom NAG params
    ({}, {**STUDIO, "negative_prompt": ""}),                 # 7 empty: no NAG
    ({}, {**STUDIO, "prompt": "talking | smiling"}),         # 8 multi-prompt: negative set, no NAG
]


def _sampler(wf):
    return next(n for n in wf.values() if n["class_type"] == "WanVideoSampler")


def _run(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(JOBS), encoding="utf-8")
    out = tmp_path / "out.json"
    proc = subprocess.run([sys.executable, str(DRIVER), str(REPO_ROOT), str(jobs), str(out)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_negative_prompt_applies_nag(tmp_path):
    data = _run(tmp_path)
    prompts = data["prompts"]
    assert len(prompts) == len(JOBS)
    assert all(set(r) == {"video"} for r in data["results"])

    for i in range(7):
        wf = prompts[i]
        assert wf["241"]["inputs"]["negative_prompt"] == NEG, f"job {i}"
        assert wf["900"]["class_type"] == "WanVideoApplyNAG", f"job {i}"
        nag = wf["900"]["inputs"]
        assert nag["original_text_embeds"] == ["241", 0]
        assert nag["nag_text_embeds"] == ["241", 1]
        assert _sampler(wf)["inputs"]["text_embeds"] == ["900", 0], f"job {i}"

    assert {k: prompts[0]["900"]["inputs"][k] for k in ("nag_scale", "nag_tau", "nag_alpha")} == \
        {"nag_scale": 11.0, "nag_tau": 2.5, "nag_alpha": 0.25}
    assert {k: prompts[6]["900"]["inputs"][k] for k in ("nag_scale", "nag_tau", "nag_alpha")} == \
        {"nag_scale": 7.0, "nag_tau": 2.0, "nag_alpha": 0.5}

    assert "900" not in prompts[7]
    assert _sampler(prompts[7])["inputs"]["text_embeds"] == ["241", 0]
    assert prompts[7]["241"]["inputs"]["negative_prompt"] != ""

    assert "900" not in prompts[8]
    assert prompts[8]["241"]["inputs"]["negative_prompt"] == NEG
    assert _sampler(prompts[8])["inputs"]["text_embeds"] == ["241", 0]
