import copy
import json
import os

import pytest

from worker_guard import vocals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = ["I2V_single.json", "I2V_multi.json", "V2V_single.json", "V2V_multi.json",
             "I2V_single_compiled.json", "I2V_multi_compiled.json"]


@pytest.mark.parametrize("name", WORKFLOWS)
def test_skip_rewires_embeds_to_load_audio(name):
    prompt = json.load(open(os.path.join(ROOT, name)))
    before = copy.deepcopy(prompt)
    n = vocals.skip_vocal_separation(prompt)
    assert n == (2 if "multi" in name else 1)
    classes = {node["class_type"] for node in prompt.values()}
    assert "MelBandRoFormerSampler" not in classes and "MelBandRoFormerModelLoader" not in classes
    for key in ("audio_1", "audio_2"):
        link = prompt["194"]["inputs"].get(key)
        if link:
            assert prompt[link[0]]["class_type"] == "LoadAudio"
            assert link == before[before["194"]["inputs"][key][0]]["inputs"]["audio"]
    # every remaining link points at a node that still exists
    for node in prompt.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in prompt


@pytest.mark.parametrize("value,skip", [(None, False), (True, False), (False, True), ("false", True),
                                        ("0", True), ("true", False)])
def test_wants_skip(value, skip):
    job = {} if value is None else {"separate_vocals": value}
    assert vocals.wants_skip(job) is skip


def test_default_workflow_untouched():
    prompt = json.load(open(os.path.join(ROOT, "I2V_single.json")))
    assert vocals.wants_skip({}) is False
    assert prompt["194"]["inputs"]["audio_1"] == ["302", 0]
