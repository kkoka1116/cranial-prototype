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

# LLM SDKs — forbidden in EVERY first-party package (invariant #1: no AI in
# the kernel/anatomy/datamodel/storage/synthesis layers; the LLM front door
# is week 5+). Add to this list rather than weakening the regex.
_LLM_SDKS = {
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

# Dependency-direction rule (CLAUDE.md package boundaries). Arrows point one
# way only: synthesis -> {datamodel, kernel, anatomy}, storage -> datamodel,
# datamodel -> {anatomy, kernel}, anatomy -> kernel, kernel standalone.
# A package may NOT import any package listed in its forbidden set.
_LAYER_FORBIDDEN: dict[str, set[str]] = {
    "kernel": _LLM_SDKS | {"anatomy", "datamodel", "storage", "synthesis"},
    "anatomy": _LLM_SDKS | {"datamodel", "storage", "synthesis"},
    "datamodel": _LLM_SDKS | {"storage", "synthesis"},
    "storage": _LLM_SDKS | {"synthesis"},
    "synthesis": set(_LLM_SDKS),
}

# Default set used by scan_file when no per-layer set is passed (and by the
# unit tests): the broadest non-kernel rule — LLM SDKs plus every layer that
# nothing below synthesis may import.
FORBIDDEN_PACKAGES = _LLM_SDKS | {"datamodel", "storage", "synthesis"}

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

_ROOT = Path(__file__).resolve().parent.parent


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def scan_file(path: Path, forbidden: set[str] | None = None) -> list[tuple[int, str]]:
    """Return list of (line_no, line) where a forbidden import appears.

    `forbidden` defaults to the broad FORBIDDEN_PACKAGES set; main() passes
    the per-layer set so the dependency-direction rule is enforced
    asymmetrically (e.g. storage MAY import datamodel but NOT synthesis).
    """
    forbidden = forbidden if forbidden is not None else FORBIDDEN_PACKAGES
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
            if module in forbidden or root in forbidden:
                line_no = text[: match.start()].count("\n") + 1
                hits.append((line_no, lines[line_no - 1]))
                break  # one hit per import line is enough

    for match in _DYNAMIC_RE.finditer(text):
        module = match.group("mod")
        root = _module_root(module)
        if module in forbidden or root in forbidden:
            line_no = text[: match.start()].count("\n") + 1
            hits.append((line_no, lines[line_no - 1]))

    return hits


def main() -> int:
    bad: list[tuple[Path, int, str]] = []
    for pkg, forbidden in _LAYER_FORBIDDEN.items():
        pkg_dir = _ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for py in sorted(pkg_dir.rglob("*.py")):
            for line_no, line in scan_file(py, forbidden):
                bad.append((py, line_no, line))
    if not bad:
        return 0
    sys.stderr.write(
        "FORBIDDEN IMPORT — LLM SDK or dependency-direction violation "
        "(invariants #1/#11, CLAUDE.md package boundaries):\n"
    )
    for path, line_no, line in bad:
        sys.stderr.write(f"  {path}:{line_no}: {line.strip()}\n")
    sys.stderr.write(
        "\nNo first-party package may import an LLM SDK, and the dependency "
        "arrows point one way only: synthesis -> {datamodel, kernel, "
        "anatomy}, storage -> datamodel, datamodel -> {anatomy, kernel}, "
        "anatomy -> kernel. A lower layer must never import a higher one.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
