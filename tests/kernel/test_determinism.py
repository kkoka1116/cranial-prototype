"""Determinism tests — invariant #2: same inputs produce bit-identical STL.

These are the golden tests called out in week-1 acceptance criterion #4.
Each example config has a recorded SHA-256 in tests/fixtures/golden/hashes.json;
this test asserts the live pipeline matches.

If the kernel intentionally changes (e.g., dependency bump), regenerate the
goldens with:

    uv run python -m kernel.generate --config examples/<name>.yaml --out /tmp/h.stl

The hash printed to stdout is the new golden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel.config import KernelConfig
from kernel.export.stl import sha256_of_mesh_stl
from kernel.generate import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_HASHES_PATH = REPO_ROOT / "tests" / "fixtures" / "golden" / "hashes.json"
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_goldens() -> dict[str, str]:
    return json.loads(GOLDEN_HASHES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "config_name",
    ["baseline", "moderate_right_plagio", "brachycephaly"],
)
def test_golden_hash_matches(config_name: str) -> None:
    config = KernelConfig.from_yaml(EXAMPLES_DIR / f"{config_name}.yaml")
    helmet, _ = run_pipeline(config)
    actual_hash = sha256_of_mesh_stl(helmet)
    expected_hash = _load_goldens()[config_name]
    assert actual_hash == expected_hash, (
        f"Hash drift for {config_name}: expected {expected_hash}, got {actual_hash}. "
        f"If this change was intentional, update tests/fixtures/golden/hashes.json."
    )


@pytest.mark.parametrize(
    "config_name",
    ["baseline", "moderate_right_plagio", "brachycephaly"],
)
def test_same_config_produces_same_hash_twice(config_name: str) -> None:
    """Run the same config twice and confirm bit-identical STL output.

    Independent of the golden hash — this catches in-process nondeterminism
    even if the goldens drift (which would be a separate failure).
    """
    config = KernelConfig.from_yaml(EXAMPLES_DIR / f"{config_name}.yaml")
    helmet1, _ = run_pipeline(config)
    helmet2, _ = run_pipeline(config)
    assert sha256_of_mesh_stl(helmet1) == sha256_of_mesh_stl(helmet2)
