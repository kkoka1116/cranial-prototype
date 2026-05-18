"""Anatomy test fixtures.

Synthetic-head scans are generated from the kernel at session scope and
written to a temp directory rather than committed as binaries (the project
convention is "commit hashes/JSON, not STLs"). Tests that need a scan file
on disk depend on `synthetic_scan_paths`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head

# Three HeadConfigs mirroring the Week 1 example helmets (head section only).
_HEAD_CONFIGS: dict[str, HeadConfig] = {
    "baseline": HeadConfig(),
    "plagio": HeadConfig(
        occipital_flat_mm=15.0,
        occipital_flat_radius_mm=55.0,
        occipital_flat_lateral_offset_mm=35.0,
    ),
    "brachy": HeadConfig(
        occipital_flat_mm=6.0,
        occipital_flat_radius_mm=55.0,
        frontal_bossing_mm=5.0,
        frontal_bossing_radius_mm=40.0,
        brachycephaly_factor=0.85,
    ),
}


@pytest.fixture(scope="session")
def head_configs() -> dict[str, HeadConfig]:
    return dict(_HEAD_CONFIGS)


@pytest.fixture(scope="session")
def synthetic_scan_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate synthetic-head STLs once per session; return name -> path."""
    out_dir = tmp_path_factory.mktemp("synthetic_scans")
    paths: dict[str, Path] = {}
    for name, cfg in _HEAD_CONFIGS.items():
        mesh = generate_test_head(cfg)
        p = out_dir / f"synthetic_{name}.stl"
        mesh.export(p)
        paths[name] = p
    return paths
