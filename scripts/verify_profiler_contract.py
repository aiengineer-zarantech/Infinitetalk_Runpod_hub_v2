# ==============================================================================
# verify_profiler_contract.py - Build-time check (Dockerfile.v22) that the INSTALLED
# ComfyUI-WanVideoWrapper still has everything comfy_profiler wraps, in the order its window
# math assumes. Reads the sources with `ast` (no import, no GPU). Exit 1 on any mismatch.
# Usage: python verify_profiler_contract.py <wrapper_dir>
# ==============================================================================
import ast
import os
import sys

STEP_BAR_DESC = "Sampling audio indices"


def _parse(wrapper_dir, rel):
    with open(os.path.join(wrapper_dir, rel), encoding="utf-8") as f:
        return ast.parse(f.read())


def _imports_from(tree, module, name, level):
    return any(
        isinstance(n, ast.ImportFrom) and n.module == module and n.level == level and any(a.name == name for a in n.names)
        for n in tree.body
    )


def _imports(tree, name, asname=None):
    return any(
        isinstance(n, ast.Import) and any(a.name == name and a.asname == asname for a in n.names) for n in tree.body
    )


def _function(tree, name):
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def _attr_calls(node, obj, attr):
    """Line numbers of calls obj.attr(...)."""
    return sorted(
        n.lineno for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
        and isinstance(n.func.value, ast.Name) and n.func.value.id == obj
    )


def _step_bar_calls(fn):
    """(line, target name) of tqdm(..., desc=f"Sampling audio indices ...") assignments."""
    out = []
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
            continue
        call = n.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "tqdm"):
            continue
        for kw in call.keywords:
            if kw.arg != "desc":
                continue
            first = kw.value.values[0] if isinstance(kw.value, ast.JoinedStr) and kw.value.values else kw.value
            if isinstance(first, ast.Constant) and str(first.value).startswith(STEP_BAR_DESC):
                target = n.targets[0].id if isinstance(n.targets[0], ast.Name) else None
                out.append((n.lineno, target))
    return out


def check(wrapper_dir):
    errors = []

    sampler = _parse(wrapper_dir, "nodes_sampler.py")
    if not _imports_from(sampler, "multitalk.multitalk_loop", "multitalk_loop", 1):
        errors.append("nodes_sampler.py: no 'from .multitalk.multitalk_loop import multitalk_loop'")
    calls = [n for n in ast.walk(sampler) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "multitalk_loop"]
    if not calls:
        errors.append("nodes_sampler.py: multitalk_loop(...) is never called by name")

    loop = _parse(wrapper_dir, os.path.join("multitalk", "multitalk_loop.py"))
    if not _imports_from(loop, "tqdm", "tqdm", 0):
        errors.append("multitalk_loop.py: no 'from tqdm import tqdm'")
    if not _imports(loop, "gc"):
        errors.append("multitalk_loop.py: no 'import gc'")
    if not _imports(loop, "comfy.model_management", "mm"):
        errors.append("multitalk_loop.py: no 'import comfy.model_management as mm'")
    fn = _function(loop, "multitalk_loop")
    if fn is None:
        errors.append("multitalk_loop.py: def multitalk_loop missing")
    else:
        bars = _step_bar_calls(fn)
        if len(bars) != 1 or bars[0][1] is None:
            errors.append(f"multitalk_loop: expected ONE 'x = tqdm(desc=f\"{STEP_BAR_DESC} ...\")', found {bars}")
        else:
            bar_line, bar = bars[0]
            updates, closes = _attr_calls(fn, bar, "update"), _attr_calls(fn, bar, "close")
            encodes, decodes = _attr_calls(fn, "vae", "encode"), _attr_calls(fn, "vae", "decode")
            if len(updates) != 1 or not updates[0] > bar_line:
                errors.append(f"multitalk_loop: expected one {bar}.update() after the bar, found lines {updates}")
            if not decodes or not closes:
                errors.append("multitalk_loop: vae.decode() or the step bar close() missing")
            elif not (bar_line < min(decodes) < max(closes)):
                errors.append("multitalk_loop: the step bar no longer closes after vae.decode (window end mark)")
            if not encodes or not min(encodes) < bar_line:
                errors.append("multitalk_loop: vae.encode() no longer runs before the step loop")
        for obj, attr in (("vae", "to"), ("mm", "soft_empty_cache"), ("gc", "collect")):
            if not _attr_calls(fn, obj, attr):
                errors.append(f"multitalk_loop: {obj}.{attr}() is never called")

    vae = _parse(wrapper_dir, os.path.join("wanvideo", "wan_video_vae.py"))
    cls = next((n for n in vae.body if isinstance(n, ast.ClassDef) and n.name == "WanVideoVAE"), None)
    if cls is None:
        errors.append("wan_video_vae.py: class WanVideoVAE missing")
    else:
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name, arg in (("encode", "videos"), ("decode", "hidden_states")):
            m = methods.get(name)
            args = [a.arg for a in m.args.args] if m else []
            if args[:2] != ["self", arg]:
                errors.append(f"WanVideoVAE.{name}: expected (self, {arg}, ...), found {args}")
        if "to" in methods:
            errors.append("WanVideoVAE defines its own to(); the profiler expects nn.Module.to")
    return errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: verify_profiler_contract.py <wrapper_dir>")
        return 2
    errors = check(argv[0])
    for e in errors:
        print(f"PROFILER CONTRACT ERROR: {e}")
    if errors:
        return 1
    print("profiler contract OK: multitalk_loop, step bar, VAE encode/decode/to, mm.soft_empty_cache, gc.collect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
