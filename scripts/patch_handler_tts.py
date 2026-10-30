# ==============================================================================
# patch_handler_tts.py - Build-time patch for Dockerfile.tts
# Inserts the 4-line TTS routing block into the base image's own /handler.py,
# right after `job_input = job.get("input", {})`. Fails (exit 1, file untouched) if
# the anchor matches 0 or >1 times, the block is already present, or the result
# is not exactly 4 added / 0 removed lines of valid Python.
# Usage: python3 patch_handler_tts.py /handler.py
# ==============================================================================
import ast
import difflib
import sys

ANCHOR = 'job_input = job.get("input", {})'
ALREADY_PRESENT_MARKERS = (
    "from tts.tts_handler import handle_tts",
    '.get("task") == "tts"',
)
BLOCK = [
    "",
    'if job_input.get("task") == "tts":',
    "    from tts.tts_handler import handle_tts",
    "    return handle_tts(job_input)",
]


def patch_source(source: str) -> str:
    """Returns the patched source, or raises ValueError explaining why it cannot patch."""
    for marker in ALREADY_PRESENT_MARKERS:
        if marker in source:
            raise ValueError(f"TTS routing block already present (found: {marker!r})")

    lines = source.splitlines(keepends=True)
    anchor_idx = [i for i, line in enumerate(lines) if line.strip() == ANCHOR]
    if len(anchor_idx) != 1:
        raise ValueError(f"Anchor line {ANCHOR!r} must match exactly once, matched {len(anchor_idx)} times")

    idx = anchor_idx[0]
    anchor_line = lines[idx]
    newline = "\r\n" if anchor_line.endswith("\r\n") else "\n"
    indent = anchor_line[: len(anchor_line) - len(anchor_line.lstrip())]
    if not anchor_line.endswith(("\r\n", "\n")):
        lines[idx] = anchor_line + newline

    block = [(indent + line if line else "") + newline for line in BLOCK]
    patched_lines = lines[: idx + 1] + block + lines[idx + 1:]
    patched = "".join(patched_lines)

    # Self-checks: valid Python, and exactly 4 added / 0 removed lines
    ast.parse(patched)
    diff = list(difflib.unified_diff(source.splitlines(), patched.splitlines(), lineterm="", n=0))
    added = [d for d in diff if d.startswith("+") and not d.startswith("+++")]
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    if len(added) != 4 or removed:
        raise ValueError(f"Unexpected diff: {len(added)} added, {len(removed)} removed (expected 4 / 0)")
    return patched


def main(argv) -> int:
    if len(argv) != 2:
        print("Usage: patch_handler_tts.py <path/to/handler.py>", file=sys.stderr)
        return 2
    path = argv[1]
    with open(path, "r", encoding="utf-8", newline="") as f:
        source = f.read()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as e:
        print(f"ERROR patch_handler_tts: {e}. {path} left unchanged.", file=sys.stderr)
        return 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(patched)
    print(f"OK patch_handler_tts: inserted 4-line TTS routing block into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
