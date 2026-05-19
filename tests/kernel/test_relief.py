"""Tests for the local relief-zone carving primitive.

The input helmet is built with the real Week-1 pipeline (synthetic head ->
``build_shell``) so the tests exercise realistic geometry.

Volume-sign note (test_relief_adds_clearance): the shell is an
``outer - inner`` manifold, so ``trimesh.Trimesh.volume`` measures the *solid
wall material*. A relief removes wall material from the cavity side, so the
relieved shell's ``volume`` is **strictly smaller** than the input's, and a
deeper relief removes strictly more (volume decreases monotonically with
``depth_mm``). That decreasing-solid-volume sign is the geometrically correct
direction for "the head gets more clearance here".
"""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config import HeadConfig, ShellConfig
from kernel.errors import GeometryError
from kernel.operations.shell import build_shell
from kernel.primitives.relief import apply_relief
from kernel.primitives.test_head import generate_test_head

# A point on the +X side of the inner cavity wall at mid-height. Verified
# (via the Week-1 pipeline) to sit on the cavity surface with ~4 mm of wall
# outboard of it, so a 2 mm relief is well within the wall.
_RELIEF_CENTER = (68.0, 0.0, 57.5)


@pytest.fixture(scope="module")
def shell():
    head = generate_test_head(HeadConfig())
    shell_mesh, _inner, _outer = build_shell(head, ShellConfig())
    return shell_mesh


def test_relief_output_is_manifold_watertight(shell) -> None:
    relieved = apply_relief(
        shell,
        center_mm=_RELIEF_CENTER,
        radius_mm=25.0,
        depth_mm=2.0,
        falloff="gaussian",
    )
    assert relieved.is_watertight
    assert relieved.is_winding_consistent
    assert relieved.is_volume


def test_relief_adds_clearance(shell) -> None:
    """Relief removes solid wall material from the cavity side.

    Sign/meaning: ``volume`` is the solid wall material of the
    ``outer - inner`` shell, so removing material makes it *strictly smaller*
    (the head-facing cavity grows -> more clearance). A deeper relief removes
    strictly more material. We also bound the change so a modest relief does
    not nuke the whole shell.
    """
    shell_vol = float(shell.volume)

    shallow = apply_relief(shell, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=1.0)
    deeper = apply_relief(shell, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=2.5)

    # Material was removed -> solid volume strictly decreased.
    assert shallow.volume < shell_vol
    assert deeper.volume < shell_vol

    # Deeper relief removes strictly more material than the shallow one.
    assert deeper.volume < shallow.volume

    # Bounded: a modest relief changed geometry measurably but did not remove
    # an unreasonable fraction of the shell.
    removed_fraction = (shell_vol - deeper.volume) / shell_vol
    assert 0.0 < removed_fraction < 0.25

    # The change is localized near _RELIEF_CENTER: the relieved mesh has new
    # vertices in the neighbourhood of the carve that the original lacked.
    center = np.array(_RELIEF_CENTER)
    orig_near = np.linalg.norm(shell.vertices - center, axis=1) < 30.0
    new_near = np.linalg.norm(deeper.vertices - center, axis=1) < 30.0
    assert new_near.sum() > orig_near.sum()


def test_relief_is_deterministic(shell) -> None:
    a = apply_relief(shell, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=2.0)
    b = apply_relief(shell, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=2.0)
    assert a.export(file_type="stl") == b.export(file_type="stl")


def test_relief_perforation_raises(shell) -> None:
    """An absurd depth far exceeding the local wall thickness must raise."""
    with pytest.raises(GeometryError, match="perforate"):
        apply_relief(
            shell,
            center_mm=_RELIEF_CENTER,
            radius_mm=25.0,
            depth_mm=100.0,
        )


def test_relief_does_not_mutate_input(shell) -> None:
    before = shell.vertices.copy()
    relieved = apply_relief(shell, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=2.0)
    # Input vertices untouched.
    assert np.array_equal(shell.vertices, before)
    # Output is a distinct object.
    assert relieved is not shell


def test_relief_rejects_invalid_input() -> None:
    """A non-volume input mesh is rejected before any geometry work."""
    import trimesh

    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(GeometryError, match="not a valid volume"):
        apply_relief(open_mesh, center_mm=_RELIEF_CENTER, radius_mm=25.0, depth_mm=2.0)


def test_relief_falloff_variants_are_watertight(shell) -> None:
    """Every supported falloff profile yields a valid watertight shell."""
    for falloff in ("gaussian", "linear", "cosine"):
        relieved = apply_relief(
            shell,
            center_mm=_RELIEF_CENTER,
            radius_mm=25.0,
            depth_mm=2.0,
            falloff=falloff,
        )
        assert relieved.is_watertight, falloff
        assert relieved.is_winding_consistent, falloff
        assert relieved.volume < shell.volume, falloff
