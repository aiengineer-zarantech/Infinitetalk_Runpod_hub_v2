# ==============================================================================
# patch_handler_guard.py - Build-time patch for Dockerfile.v18
# Inserts the 2-line worker_guard install block into the base image's own /handler.py,
# directly before the module-level `runpod.serverless.start(` call. Fails (exit 1, file
# untouched) if the anchor matches 0 or >1 times, the block is already present, the file
# has no top-level `handler` / `get_videos` functions, or the result is not exactly
# 2 added / 0 removed lines of valid Python.
# Usage: python3 patch_handler_guard.py /handler.py
# ==============================================================================
import ast
import difflib
import sys

ANCHOR = "runpod.serverless.start("
ALREADY_PRESENT_MARKERS = ("worker_guard.job_guard", "install_job_guard")
REQUIRED_FUNCTIONS = ("handler", "get_videos")
BLOCK = [
    "from worker_guard.job_guard import install as install_job_guard",
    "handler, get_videos = install_job_guard(handler, get_videos)",
]


def patch_source(source: str) -> str:
    """Returns the patched source, or raises ValueError explaining why it cannot patch."""
    for marker in ALREADY_PRESENT_MARKERS:
        if marker in source:
            raise ValueError(f"worker_guard block already present (found: {marker!r})")

    tree = ast.parse(source)
    top_level = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in REQUIRED_FUNCTIONS if name not in top_level]
    if missing:
        raise ValueError(f"top-level function(s) {missing} not found")

    lines = source.splitlines(keepends=True)
    # Module level only: the anchor must start at column 0
    anchor_idx = [i for i, line in enumerate(lines) if line.startswith(ANCHOR)]
    if len(anchor_idx) != 1:
        raise ValueError(f"Anchor line starting with {ANCHOR!r} must match exactly once, matched {len(anchor_idx)} times")

    idx = anchor_idx[0]
    newline = "\r\n" if lines[idx].endswith("\r\n") else "\n"
    block = [line + newline for line in BLOCK]
    patched = "".join(lines[:idx] + block + lines[idx:])

    # Self-checks: valid Python, and exactly 2 added / 0 removed lines
    ast.parse(patched)
    diff = list(difflib.unified_diff(source.splitlines(), patched.splitlines(), lineterm="", n=0))
    added = [d for d in diff if d.startswith("+") and not d.startswith("+++")]
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    if len(added) != len(BLOCK) or removed:
        raise ValueError(f"Unexpected diff: {len(added)} added, {len(removed)} removed (expected {len(BLOCK)} / 0)")
    return patched


def main(argv) -> int:
    if len(argv) != 2:
        print("Usage: patch_handler_guard.py <path/to/handler.py>", file=sys.stderr)
        return 2
    path = argv[1]
    with open(path, "r", encoding="utf-8", newline="") as f:
        source = f.read()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as e:
        print(f"ERROR patch_handler_guard: {e}. {path} left unchanged.", file=sys.stderr)
        return 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(patched)
    print(f"OK patch_handler_guard: inserted 2-line worker_guard block into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
