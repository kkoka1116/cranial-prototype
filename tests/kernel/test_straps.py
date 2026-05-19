"""Tests for the strap-mount tab-fusion primitive.

The input helmet is built with the real Week-1 pipeline (synthetic head ->
``build_shell`` -> ``apply_trim``) so the tests exercise realistic geometry
with a genuine lower trim edge.

Volume-sign note (test_strap_adds_material): the shell is an ``outer - inner``
manifold, so ``trimesh.Trimesh.volume`` measures the *solid wall material*. A
strap mount boolean-*unions* a solid tab onto the shell, which only ever adds
material, so the strapped shell's ``volume`` is **strictly larger** than the
input's, and a bigger tab (``strap_width_mm`` / ``reinforcement_radius_mm``)
adds strictly more. That increasing-solid-volume sign is the geometrically
correct direction for "extra material was fused on".

Input-component note: ``apply_trim`` leaves a tiny (~12 mm^3) disconnected
stub on the Z-axis at the bottom of the trimmed helmet — an artifact of its
1 mm inner-radius cutting body, unrelated to straps. We therefore drive the
tests with the *largest* (by absolute volume) watertight component, which is
the real helmet body and is itself a single-component, watertight volume.
Selecting that component is a legitimate, purely additive test convenience
(no kernel source is modified).
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from kernel.config import HeadConfig, ShellConfig, TrimConfig, TrimControlPoint
from kernel.errors import GeometryError
from kernel.operations.shell import build_shell
from kernel.operations.trim import apply_trim
from kernel.primitives.straps import apply_strap_mount
from kernel.primitives.test_head import generate_test_head

# A helmet vertex with near-minimal Z on the +X side: this sits right on the
# lower trim edge (z ~= 34.6 mm) on the patient's right, exactly where a
# retention strap mount belongs.
_STRAP_ANCHOR = (67.0, 5.5, 34.6)


@pytest.fixture(scope="module")
def helmet() -> trimesh.Trimesh:
    head = generate_test_head(HeadConfig())
    shell_mesh, _inner, _outer = build_shell(head, ShellConfig())
    trim = TrimConfig(
        control_points=[
            TrimControlPoint(azimuth_deg=a, elevation_deg=-10.0) for a in (0.0, 90.0, 180.0, 270.0)
        ]
    )
    trimmed = apply_trim(shell_mesh, trim)
    # Use the real helmet body (largest watertight component); apply_trim
    # leaves a tiny disconnected Z-axis stub that is not strap-related.
    body = max(trimmed.split(only_watertight=False), key=lambda c: abs(c.volume))
    assert body.is_watertight and body.is_winding_consistent and body.is_volume
    assert len(body.split(only_watertight=False)) == 1
    return body


def test_strap_output_manifold_watertight(helmet: trimesh.Trimesh) -> None:
    result = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    assert result.is_watertight
    assert result.is_winding_consistent
    assert result.is_volume


def test_strap_adds_material(helmet: trimesh.Trimesh) -> None:
    """A strap union only adds material, so solid volume strictly grows.

    A larger ``strap_width_mm`` (and, independently, a larger
    ``reinforcement_radius_mm``) builds a bigger tab and therefore adds
    strictly more material than the smaller one.
    """
    base_vol = float(helmet.volume)

    default = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    assert default.volume > base_vol

    narrow = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, strap_width_mm=15.0)
    wide = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, strap_width_mm=30.0)
    assert narrow.volume > base_vol
    assert wide.volume > narrow.volume

    small_r = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, reinforcement_radius_mm=6.0)
    big_r = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, reinforcement_radius_mm=12.0)
    assert small_r.volume > base_vol
    assert big_r.volume > small_r.volume


def test_strap_is_connected(helmet: trimesh.Trimesh) -> None:
    """The tab fuses into one solid — not a floating appendage."""
    result = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    assert len(result.split(only_watertight=False)) == 1


def test_strap_is_deterministic(helmet: trimesh.Trimesh) -> None:
    a = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    b = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    assert a.export(file_type="stl") == b.export(file_type="stl")


def test_strap_angle_changes_orientation(helmet: trimesh.Trimesh) -> None:
    """``angle_deg`` tilts the tab, so the exported mesh differs."""
    a = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, angle_deg=10.0)
    b = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, angle_deg=50.0)
    assert a.export(file_type="stl") != b.export(file_type="stl")
    assert a.is_watertight and b.is_watertight
    assert len(a.split(only_watertight=False)) == 1
    assert len(b.split(only_watertight=False)) == 1


def test_strap_rejects_bad_params(helmet: trimesh.Trimesh) -> None:
    with pytest.raises(GeometryError, match="strap_width_mm"):
        apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, strap_width_mm=0.0)
    with pytest.raises(GeometryError, match="reinforcement_radius_mm"):
        apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, reinforcement_radius_mm=0.0)
    with pytest.raises(GeometryError, match="angle_deg"):
        apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR, angle_deg=120.0)


def test_strap_rejects_invalid_input() -> None:
    """A non-volume input mesh is rejected before any geometry work."""
    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(GeometryError, match="not a valid volume"):
        apply_strap_mount(open_mesh, anchor_mm=_STRAP_ANCHOR)


def test_strap_does_not_mutate_input(helmet: trimesh.Trimesh) -> None:
    before = helmet.vertices.copy()
    result = apply_strap_mount(helmet, anchor_mm=_STRAP_ANCHOR)
    # Input vertices untouched.
    assert np.array_equal(helmet.vertices, before)
    # Output is a distinct object.
    assert result is not helmet
