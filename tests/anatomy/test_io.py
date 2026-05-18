"""Tests for scan loading and scale inference."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from anatomy.errors import AnatomyError
from anatomy.io import infer_and_correct_scale, load_scan


def test_load_stl_mm(synthetic_scan_paths: dict[str, Path]) -> None:
    mesh = load_scan(synthetic_scan_paths["baseline"])
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0
    # Synthetic baseline head is ~165 mm long; longest extent in mm range.
    assert 140.0 <= max(mesh.extents) <= 200.0


@pytest.mark.parametrize("fmt", ["ply", "obj"])
def test_load_ply_and_obj(synthetic_scan_paths: dict[str, Path], tmp_path: Path, fmt: str) -> None:
    src = trimesh.load(synthetic_scan_paths["baseline"], process=False)
    p = tmp_path / f"head.{fmt}"
    src.export(p)
    mesh = load_scan(p)
    assert len(mesh.faces) > 0
    assert 140.0 <= max(mesh.extents) <= 200.0


def test_scale_inference_meters(synthetic_scan_paths: dict[str, Path], tmp_path: Path) -> None:
    src = trimesh.load(synthetic_scan_paths["baseline"], process=False)
    src.apply_scale(0.001)  # mm -> m
    p = tmp_path / "head_m.stl"
    src.export(p)
    mesh = load_scan(p)
    # Auto-converted back to mm.
    assert 140.0 <= max(mesh.extents) <= 200.0


def test_scale_inference_inches(synthetic_scan_paths: dict[str, Path], tmp_path: Path) -> None:
    src = trimesh.load(synthetic_scan_paths["baseline"], process=False)
    src.apply_scale(1.0 / 25.4)  # mm -> inch
    p = tmp_path / "head_in.stl"
    src.export(p)
    mesh = load_scan(p)
    assert 140.0 <= max(mesh.extents) <= 200.0


def test_absurd_scale_raises(synthetic_scan_paths: dict[str, Path], tmp_path: Path) -> None:
    src = trimesh.load(synthetic_scan_paths["baseline"], process=False)
    src.apply_scale(1000.0)  # 165 m -> absurd
    p = tmp_path / "head_huge.stl"
    src.export(p)
    with pytest.raises(AnatomyError, match="Cannot infer scan scale"):
        load_scan(p)


def test_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "scan.3mf"
    p.write_text("not a real mesh", encoding="utf-8")
    with pytest.raises(AnatomyError, match="unsupported scan format"):
        load_scan(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AnatomyError, match="not found"):
        load_scan(tmp_path / "nope.stl")


def test_infer_scale_returns_factor(synthetic_scan_paths: dict[str, Path]) -> None:
    src = trimesh.load(synthetic_scan_paths["baseline"], process=False)
    src.apply_scale(0.001)
    _mesh, units, factor = infer_and_correct_scale(src)
    assert units == "m"
    assert factor == 1000.0
