# ==============================================================================
# make_compiled_workflows.py - Writes I2V_single_compiled.json and I2V_multi_compiled.json
# The ONLY change versus the originals: node 122 (WanVideoModelLoader) gets the input
# "compile_args": ["177", 0], connecting the wrapper's own WanVideoTorchCompileSettings
# node 177 that the originals already contain (unconnected). The originals are not modified.
# Usage (from the worker repo): python scripts/make_compiled_workflows.py [--check]
#   --check: verify the existing compiled files instead of writing them (exit 1 on mismatch)
# ==============================================================================
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from worker_guard import accel  # noqa: E402

LINK = [accel.COMPILE_NODE, 0]


def build(original):
    loader = original[accel.LOADER_NODE]
    compile_node = original.get(accel.COMPILE_NODE, {})
    if loader.get("class_type") != "WanVideoModelLoader":
        raise ValueError(f"node {accel.LOADER_NODE} is {loader.get('class_type')!r}, expected WanVideoModelLoader")
    if compile_node.get("class_type") != "WanVideoTorchCompileSettings":
        raise ValueError(f"node {accel.COMPILE_NODE} is {compile_node.get('class_type')!r}, expected WanVideoTorchCompileSettings")
    if "compile_args" in loader.get("inputs", {}):
        raise ValueError("original already has compile_args")
    compiled = json.loads(json.dumps(original))
    compiled[accel.LOADER_NODE]["inputs"]["compile_args"] = LINK
    new_nodes, new_inputs = accel.workflow_delta(original, compiled)
    assert new_nodes == {} and new_inputs == {accel.LOADER_NODE: {"compile_args": LINK}}
    return compiled


def main(argv):
    check = "--check" in argv
    ok = True
    for original_name, compiled_name in accel.COMPILED_WORKFLOWS.items():
        original = json.loads((REPO_ROOT / original_name).read_text(encoding="utf-8"))
        expected = build(original)
        target = REPO_ROOT / compiled_name
        if check:
            same = target.exists() and json.loads(target.read_text(encoding="utf-8")) == expected
            print(f"{'OK  ' if same else 'DIFF'} {compiled_name}")
            ok &= same
        else:
            target.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            print(f"wrote {compiled_name} (= {original_name} + node {accel.LOADER_NODE} compile_args -> node {accel.COMPILE_NODE})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
