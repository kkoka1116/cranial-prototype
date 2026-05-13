"""Tests for the validation suite."""

from __future__ import annotations

import pytest
import trimesh
from pydantic import ValidationError

from kernel.config import HeadConfig, ShellConfig
from kernel.operations.shell import build_shell
from kernel.primitives.test_head import generate_test_head
from kernel.validation.geometry import (
    check_bbox_within,
    check_mass,
    check_min_wall_thickness,
    estimate_mass_g,
)
from kernel.validation.topology import (
    ValidationResult,
    check_manifold,
    check_watertight,
)


def _make_open_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )


def test_check_manifold_passes_on_sphere() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3)
    result = check_manifold(sphere)
    assert result.passed
    assert result.metric_name == "manifold"


def test_check_manifold_fails_on_open_mesh() -> None:
    result = check_manifold(_make_open_mesh())
    assert not result.passed


def test_check_watertight_passes_on_sphere() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3)
    result = check_watertight(sphere)
    assert result.passed
    assert result.metric_name == "watertight"


def test_check_watertight_fails_on_open_mesh() -> None:
    result = check_watertight(_make_open_mesh())
    assert not result.passed


def test_check_bbox_within_passes_and_fails() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    assert check_bbox_within(sphere, max_extent_mm=50.0).passed
    fail = check_bbox_within(sphere, max_extent_mm=5.0)
    assert not fail.passed
    assert fail.actual_value == pytest.approx(20.0, abs=0.01)


def test_estimate_mass_g_uses_density() -> None:
    # 100x100x100 mm cube = 1e6 mm³ = 1000 cm³.
    # At 1.27 g/cm³ → 1270 g.
    cube = trimesh.creation.box(extents=(100, 100, 100))
    mass = estimate_mass_g(cube, density_g_per_cm3=1.27)
    assert mass == pytest.approx(1270.0, rel=0.001)


def test_check_mass_passes_under_limit() -> None:
    cube = trimesh.creation.box(extents=(10, 10, 10))  # 1 cm³ at 1.27 g/cm³ = 1.27g
    res = check_mass(cube, density_g_per_cm3=1.27, max_mass_g=10.0)
    assert res.passed
    assert res.actual_value == pytest.approx(1.27, rel=0.001)


def test_check_mass_fails_over_limit() -> None:
    cube = trimesh.creation.box(extents=(100, 100, 100))  # 1270g
    res = check_mass(cube, density_g_per_cm3=1.27, max_mass_g=250.0)
    assert not res.passed


def test_check_min_wall_thickness_on_baseline_shell() -> None:
    head = generate_test_head(HeadConfig())
    cfg = ShellConfig(relief_mm=3.0, wall_thickness_mm=4.0)
    _shell, inner, outer = build_shell(head, cfg)
    res = check_min_wall_thickness(
        inner_surface_mesh=inner,
        outer_surface_mesh=outer,
        min_thickness_mm=3.0,
    )
    assert res.passed
    assert res.actual_value == pytest.approx(4.0, abs=0.1)


def test_check_min_wall_thickness_fails_when_too_thin() -> None:
    head = generate_test_head(HeadConfig())
    cfg = ShellConfig(relief_mm=3.0, wall_thickness_mm=2.0)
    _shell, inner, outer = build_shell(head, cfg)
    res = check_min_wall_thickness(
        inner_surface_mesh=inner,
        outer_surface_mesh=outer,
        min_thickness_mm=3.0,
    )
    assert not res.passed


def test_check_min_wall_thickness_rejects_zero_limit() -> None:
    head = generate_test_head(HeadConfig(subdivisions=3))
    _shell, inner, outer = build_shell(head, ShellConfig())
    with pytest.raises(ValueError):
        check_min_wall_thickness(inner, outer, min_thickness_mm=0.0)


def test_validation_result_is_frozen() -> None:
    r = ValidationResult(
        metric_name="x", passed=True, actual_value=1.0,
        limit_value=1.0, message="ok",
    )
    with pytest.raises(ValidationError):
        r.passed = False  # type: ignore[misc]
