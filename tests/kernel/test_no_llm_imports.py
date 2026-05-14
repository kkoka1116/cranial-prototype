"""Enforce invariant #1: no LLM SDK imports under kernel/.

This duplicates the pre-commit hook so CI catches the violation even if
the hook isn't installed locally.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_no_llm_imports.py"


def test_no_llm_imports_under_kernel() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Forbidden LLM imports detected under kernel/:\n{result.stderr}"


def test_script_detects_forbidden_import(tmp_path: Path) -> None:
    """Synthetic test: a file that imports anthropic should be flagged."""
    # Mock kernel/ tree with a forbidden import, run the script against it.
    fake_kernel = tmp_path / "kernel"
    fake_kernel.mkdir()
    (fake_kernel / "bad.py").write_text("import anthropic\n", encoding="utf-8")

    # Copy the script and adjust KERNEL_DIR
    import scripts.check_no_llm_imports as m

    bad_files = m.scan_file(fake_kernel / "bad.py")
    assert len(bad_files) == 1
    assert "anthropic" in bad_files[0][1]


def test_script_allows_legitimate_imports(tmp_path: Path) -> None:
    fake = tmp_path / "ok.py"
    fake.write_text(
        "from pathlib import Path\nimport numpy as np\nimport trimesh\n",
        encoding="utf-8",
    )
    import scripts.check_no_llm_imports as m

    assert m.scan_file(fake) == []
