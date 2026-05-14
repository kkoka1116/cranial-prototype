"""Pre-commit hook: forbid LLM SDK imports under kernel/.

Invariant #1 (CLAUDE.md): the kernel produces geometry; the LLM produces
parameters. No LLM call exists in any module under `kernel/`. This script
scans every .py file under kernel/ and fails the commit if it finds any
import of an LLM SDK.

We use a simple regex scan rather than a full AST walk — it's good enough
to catch the cases this is meant to prevent (someone curl-pasting an
example) and runs in milliseconds.

Exit codes:
    0 - no LLM imports found
    1 - one or more LLM imports found (commit blocked)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Packages we forbid. Keep this list narrow but exhaustive for current
# LLM SDKs. Add to this list rather than weakening the regex.
FORBIDDEN_PACKAGES = {
    "anthropic",
    "openai",
    "cohere",
    "google.generativeai",
    "vertexai",
    "litellm",
    "instructor",
    "llama_index",
    "langchain",
}

# Catches:
#   import X                         (and `import X as foo`)
#   import X.Y
#   import X, Y, Z                   (comma list — any forbidden module hits)
#   from X import ...
#   from X.Y import ...
# The `imp` capture allows commas / spaces / `as` aliases so we can split it
# downstream and check every imported name on the line.
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[a-zA-Z_][\w\.]*)" r"|import\s+(?P<imp>[a-zA-Z_][\w\.,\s]*))",
    flags=re.MULTILINE,
)
# Catches dynamic imports like __import__("anthropic") or
# importlib.import_module("openai.types").
_DYNAMIC_RE = re.compile(
    r"""(?:__import__|importlib\.import_module)\s*\(\s*['"]""" r"""(?P<mod>[a-zA-Z_][\w\.]*)['"]"""
)

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernel"


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, line) where forbidden imports appear."""
    hits: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for match in _IMPORT_RE.finditer(text):
        candidates: list[str] = []
        if match.group("from"):
            candidates.append(match.group("from"))
        if match.group("imp"):
            # `import a, b as c, d.e` → split on commas, strip ` as alias`, keep first token.
            for piece in match.group("imp").split(","):
                token = piece.strip().split()[0] if piece.strip() else ""
                if token:
                    candidates.append(token)
        for module in candidates:
            root = _module_root(module)
            if module in FORBIDDEN_PACKAGES or root in FORBIDDEN_PACKAGES:
                line_no = text[: match.start()].count("\n") + 1
                hits.append((line_no, lines[line_no - 1]))
                break  # one hit per import line is enough

    for match in _DYNAMIC_RE.finditer(text):
        module = match.group("mod")
        root = _module_root(module)
        if module in FORBIDDEN_PACKAGES or root in FORBIDDEN_PACKAGES:
            line_no = text[: match.start()].count("\n") + 1
            hits.append((line_no, lines[line_no - 1]))

    return hits


def main() -> int:
    if not KERNEL_DIR.is_dir():
        # Nothing to scan — treat as success.
        return 0
    bad: list[tuple[Path, int, str]] = []
    for py in sorted(KERNEL_DIR.rglob("*.py")):
        for line_no, line in scan_file(py):
            bad.append((py, line_no, line))
    if not bad:
        return 0
    sys.stderr.write("FORBIDDEN LLM IMPORT under kernel/ (invariant #1, see CLAUDE.md):\n")
    for path, line_no, line in bad:
        sys.stderr.write(f"  {path}:{line_no}: {line.strip()}\n")
    sys.stderr.write(
        "\nThe kernel must not import LLM SDKs. If you genuinely need this "
        "in another layer, move it OUT of kernel/.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
