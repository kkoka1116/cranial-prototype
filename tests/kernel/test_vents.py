"""Tests for the ventilation-pattern primitive (kernel.primitives.vents).

Strictly additive Week-4-permitted primitive: builds a realistic helmet
shell from the week-1 synthetic head + shell, then exercises
apply_vent_pattern for the manifold/watertight, material-removal,
determinism, input-validation, and purity contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from kernel.config import HeadConfig, ShellConfig
from kernel.errors import GeometryError
from kernel.operations.shell import build_shell
from kernel.primitives.test_head import generate_test_head
from kernel.primitives.vents import apply_vent_pattern


def _shell() -> trimesh.Trimesh:
    head = generate_test_head(HeadConfig())
    shell, _i, _o = build_shell(head, ShellConfig())
    return shell


def _dome_center(shell: trimesh.Trimesh) -> tuple[float, float, float]:
    """Pick the shell vertex of maximum Z — a point on the upper dome."""
    top = shell.vertices[int(np.argmax(shell.vertices[:, 2]))]
    return (float(top[0]), float(top[1]), float(top[2]))


def test_vent_output_manifold_watertight() -> None:
    shell = _shell()
    center = _dome_center(shell)
    vented = apply_vent_pattern(
        shell,
        center,
        extent_mm=30.0,
        feature_size_mm=6.0,
        open_area_fraction=0.15,
    )
    assert vented.is_watertight
    assert vented.is_winding_consistent
    assert vented.is_volume
    # A new object, not the input.
    assert vented is not shell


def test_vent_removes_material() -> None:
    shell = _shell()
    center = _dome_center(shell)

    vented_small = apply_vent_pattern(
        shell, center, extent_mm=30.0, feature_size_mm=6.0, open_area_fraction=0.10
    )
    vented_large = apply_vent_pattern(
        shell, center, extent_mm=30.0, feature_size_mm=6.0, open_area_fraction=0.30
    )

    # Vented solid (= wall material) volume is below the intact shell.
    assert vented_small.volume < shell.volume
    assert vented_large.volume < shell.volume
    # More open area removes strictly more material.
    assert vented_large.volume < vented_small.volume


def test_vent_is_deterministic() -> None:
    shell = _shell()
    center = _dome_center(shell)

    a = apply_vent_pattern(
        shell, center, extent_mm=30.0, feature_size_mm=6.0, open_area_fraction=0.15
    )
    b = apply_vent_pattern(
        shell, center, extent_mm=30.0, feature_size_mm=6.0, open_area_fraction=0.15
    )
    assert a.export(file_type="stl") == b.export(file_type="stl")


def test_vent_rejects_bad_params() -> None:
    shell = _shell()
    center = _dome_center(shell)

    with pytest.raises(GeometryError):
        apply_vent_pattern(shell, center, extent_mm=30.0, open_area_fraction=0.0)
    with pytest.raises(GeometryError):
        apply_vent_pattern(shell, center, extent_mm=30.0, open_area_fraction=0.95)
    with pytest.raises(GeometryError):
        apply_vent_pattern(shell, center, extent_mm=30.0, feature_size_mm=0.0)
    with pytest.raises(GeometryError):
        apply_vent_pattern(shell, center, extent_mm=30.0, pattern="circular")
    with pytest.raises(GeometryError):
        apply_vent_pattern(shell, center, extent_mm=0.0)


def test_vent_does_not_mutate_input() -> None:
    shell = _shell()
    center = _dome_center(shell)
    verts_before = shell.vertices.copy()

    vented = apply_vent_pattern(
        shell, center, extent_mm=30.0, feature_size_mm=6.0, open_area_fraction=0.15
    )

    assert np.array_equal(shell.vertices, verts_before)
    assert vented is not shell


def test_vent_center_controls_location() -> None:
    """The holes land near the requested center — moving the center moves
    the removed material, so two different centers produce different meshes."""
    shell = _shell()
    top = _dome_center(shell)
    # A second patch offset laterally on the dome.
    side = (top[0] + 25.0, top[1], top[2] - 10.0)

    a = apply_vent_pattern(shell, top, extent_mm=20.0, feature_size_mm=6.0, open_area_fraction=0.15)
    b = apply_vent_pattern(
        shell, side, extent_mm=20.0, feature_size_mm=6.0, open_area_fraction=0.15
    )
    assert a.export(file_type="stl") != b.export(file_type="stl")
