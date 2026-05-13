"""Shared test fixtures.

Pipeline runs are cached at module scope so the heavy boolean ops don't
re-run for every test.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path. uv-built editable installs sometimes
# don't surface to pytest's import machinery, depending on the install layout.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest  # noqa: E402

from kernel.config import KernelConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(scope="session")
def baseline_config() -> KernelConfig:
    return KernelConfig.from_yaml(EXAMPLES_DIR / "baseline.yaml")


@pytest.fixture(scope="session")
def plagio_config() -> KernelConfig:
    return KernelConfig.from_yaml(EXAMPLES_DIR / "moderate_right_plagio.yaml")


@pytest.fixture(scope="session")
def brachy_config() -> KernelConfig:
    return KernelConfig.from_yaml(EXAMPLES_DIR / "brachycephaly.yaml")
