# ==============================================================================
# verify_wrapper_contract.py - Build-time check (Dockerfile.v20) that the compiled workflows
# and FP8_FAST use node/input names that exist in the INSTALLED ComfyUI-WanVideoWrapper.
# Reads nodes_model_loading.py with `ast` (no import, no GPU). Exit 1 on any mismatch.
# Usage: python verify_wrapper_contract.py <wrapper_dir> <original.json>=<compiled.json> [...]
# ==============================================================================
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker_guard import accel  # noqa: E402


def _class_input_types(tree, class_name):
    """Returns {"required": {name: ast tuple}, "optional": {...}} from the class's INPUT_TYPES return dict."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                            out = {}
                            for key, value in zip(stmt.value.keys, stmt.value.values):
                                if isinstance(key, ast.Constant) and isinstance(value, ast.Dict):
                                    out[key.value] = {
                                        k.value: v for k, v in zip(value.keys, value.values) if isinstance(k, ast.Constant)
                                    }
                            return out
    raise ValueError(f"class {class_name} with a literal INPUT_TYPES dict not found")


def _class_attr(tree, class_name, attr):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign) and any(getattr(t, "id", None) == attr for t in item.targets):
                    return ast.literal_eval(item.value)
    return None


def _first(spec):
    """First element of an input spec tuple, as a Python literal (list of options or type name)."""
    if not isinstance(spec, ast.Tuple) or not spec.elts:
        raise ValueError("input spec is not a tuple")
    return ast.literal_eval(spec.elts[0])


def check(wrapper_dir, pairs):
    errors = []
    path = os.path.join(wrapper_dir, "nodes_model_loading.py")
    tree = ast.parse(open(path, encoding="utf-8").read())

    compile_inputs = _class_input_types(tree, "WanVideoTorchCompileSettings")
    loader_inputs = _class_input_types(tree, "WanVideoModelLoader")
    compile_names = {**compile_inputs.get("required", {}), **compile_inputs.get("optional", {})}

    if _class_attr(tree, "WanVideoTorchCompileSettings", "RETURN_TYPES") != ("WANCOMPILEARGS",):
        errors.append("WanVideoTorchCompileSettings.RETURN_TYPES is not ('WANCOMPILEARGS',)")
    spec = loader_inputs.get("optional", {}).get("compile_args")
    if spec is None or _first(spec) != "WANCOMPILEARGS":
        errors.append("WanVideoModelLoader has no optional compile_args input of type WANCOMPILEARGS")
    quant = loader_inputs.get("required", {}).get("quantization")
    options = _first(quant) if quant is not None else []
    for value in (accel.FP8_FROM, accel.FP8_TO):
        if value not in options:
            errors.append(f"WanVideoModelLoader quantization has no option {value!r}")

    for original_path, compiled_path in pairs:
        original = json.load(open(original_path, encoding="utf-8"))
        compiled = json.load(open(compiled_path, encoding="utf-8"))
        new_nodes, new_inputs = accel.workflow_delta(original, compiled)
        if new_nodes or new_inputs != {accel.LOADER_NODE: {"compile_args": [accel.COMPILE_NODE, 0]}}:
            errors.append(f"{compiled_path}: differs from {original_path} by more than node 122 compile_args")
        node = compiled.get(accel.COMPILE_NODE, {})
        if node.get("class_type") != "WanVideoTorchCompileSettings":
            errors.append(f"{compiled_path}: node {accel.COMPILE_NODE} is not WanVideoTorchCompileSettings")
        if compiled.get(accel.LOADER_NODE, {}).get("class_type") != "WanVideoModelLoader":
            errors.append(f"{compiled_path}: node {accel.LOADER_NODE} is not WanVideoModelLoader")
        for name, value in node.get("inputs", {}).items():
            if name not in compile_names:
                errors.append(f"{compiled_path}: node {accel.COMPILE_NODE} input {name!r} is not in the installed INPUT_TYPES")
                continue
            allowed = _first(compile_names[name])
            if isinstance(allowed, list) and value not in allowed:
                errors.append(f"{compiled_path}: {name}={value!r} not in installed options {allowed}")
        missing = [n for n in compile_inputs.get("required", {}) if n not in node.get("inputs", {})]
        if missing:
            errors.append(f"{compiled_path}: node {accel.COMPILE_NODE} misses required inputs {missing}")
        print(f"checked {compiled_path}: node {accel.COMPILE_NODE} inputs {sorted(node.get('inputs', {}))}")
    return errors


def main(argv):
    if len(argv) < 2:
        print(__doc__ or "usage: verify_wrapper_contract.py <wrapper_dir> <original>=<compiled> ...", file=sys.stderr)
        return 2
    pairs = [tuple(arg.split("=", 1)) for arg in argv[1:]]
    errors = check(argv[0], pairs)
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    print("wrapper contract OK" if not errors else f"wrapper contract FAILED ({len(errors)} error(s))")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
