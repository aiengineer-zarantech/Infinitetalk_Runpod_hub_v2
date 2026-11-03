"""separate_vocals=false: skip MelBandRoFormer vocal separation for clean audio (Voice Lab TTS).

Every Wav2VecEmbeds audio input that comes from a MelBandRoFormerSampler is rewired to that sampler's own audio
input (the LoadAudio node), and the now unused sampler / loader nodes are removed from the prompt.
Default (key missing or true) leaves the workflow unchanged.
"""

SAMPLER = "MelBandRoFormerSampler"
LOADER = "MelBandRoFormerModelLoader"


def wants_skip(job_input):
    value = job_input.get("separate_vocals", True)
    if isinstance(value, str):
        return value.strip().lower() in ("0", "false", "no", "off")
    return value is False


def skip_vocal_separation(prompt):
    """Rewire in place. Returns the number of audio inputs rewired (0 = nothing to do)."""
    samplers = {k: n for k, n in prompt.items() if n.get("class_type") == SAMPLER}
    rewired = 0
    for node in prompt.values():
        if node.get("class_type") in (SAMPLER, LOADER):
            continue
        inputs = node.get("inputs", {})
        for name, link in list(inputs.items()):
            if isinstance(link, list) and len(link) == 2 and str(link[0]) in samplers and link[1] == 0:
                inputs[name] = list(samplers[str(link[0])]["inputs"]["audio"])
                rewired += 1
    if not rewired:
        return 0
    for k in samplers:
        del prompt[k]
    still_used = {str(v[0]) for n in prompt.values() for v in n.get("inputs", {}).values()
                  if isinstance(v, list) and len(v) == 2}
    for k in [k for k, n in prompt.items() if n.get("class_type") == LOADER and k not in still_used]:
        del prompt[k]
    return rewired
