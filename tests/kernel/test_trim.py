"""Tests for the trim-line operation."""

from __future__ import annotations

import pytest

from kernel.config import HeadConfig, ShellConfig, TrimConfig, TrimControlPoint
from kernel.operations.shell import build_shell
from kernel.operations.trim import (
    _interpolate_elevation,
    _sorted_control_points,
    apply_trim,
)
from kernel.primitives.test_head import generate_test_head


def _flat_trim(elevation_deg: float) -> TrimConfig:
    return TrimConfig(
        control_points=[
            TrimControlPoint(azimuth_deg=0.0, elevation_deg=elevation_deg),
            TrimControlPoint(azimuth_deg=90.0, elevation_deg=elevation_deg),
            TrimControlPoint(azimuth_deg=180.0, elevation_deg=elevation_deg),
            TrimControlPoint(azimuth_deg=270.0, elevation_deg=elevation_deg),
        ]
    )


def test_sorted_control_points_normalizes_negatives() -> None:
    pts = [
        TrimControlPoint(azimuth_deg=-90.0, elevation_deg=0.0),
        TrimControlPoint(azimuth_deg=0.0, elevation_deg=10.0),
        TrimControlPoint(azimuth_deg=180.0, elevation_deg=20.0),
        TrimControlPoint(azimuth_deg=359.0, elevation_deg=30.0),
    ]
    sorted_pts = _sorted_control_points(pts)
    azimuths = [p.azimuth_deg for p in sorted_pts]
    assert azimuths == sorted(azimuths)
    assert 0.0 <= min(azimuths) and max(azimuths) < 360.0


def test_elevation_interpolates_linearly() -> None:
    pts = _sorted_control_points(
        [
            TrimControlPoint(azimuth_deg=0.0, elevation_deg=0.0),
            TrimControlPoint(azimuth_deg=90.0, elevation_deg=10.0),
            TrimControlPoint(azimuth_deg=180.0, elevation_deg=0.0),
            TrimControlPoint(azimuth_deg=270.0, elevation_deg=-10.0),
        ]
    )
    assert _interpolate_elevation(pts, 0.0) == pytest.approx(0.0)
    assert _interpolate_elevation(pts, 45.0) == pytest.approx(5.0)
    assert _interpolate_elevation(pts, 90.0) == pytest.approx(10.0)
    assert _interpolate_elevation(pts, 135.0) == pytest.approx(5.0)


def test_apply_trim_removes_bottom_and_keeps_top() -> None:
    """A flat trim at elevation 0° should remove the lower half, not the upper."""
    head = generate_test_head(HeadConfig())
    shell, _inner, _outer = build_shell(head, ShellConfig())
    cfg = _flat_trim(0.0)
    trimmed = apply_trim(shell, cfg)
    assert trimmed.is_winding_consistent
    assert trimmed.is_volume
    assert trimmed.is_watertight
    # Volume reduction: trim took out a substantial chunk.
    assert trimmed.volume < shell.volume * 0.7
    # Directional assertion: the volume-weighted centroid moves clearly upward
    # after cutting the bottom. The shell's centroid sits near z=0 (helmet
    # wraps a head ellipsoid centered on origin); removing the lower half
    # should push the centroid well above z=0.
    assert trimmed.centroid[2] > shell.centroid[2] + 10.0


def test_apply_trim_rejects_invalid_input() -> None:
    """apply_trim should reject inputs that aren't valid volumes."""
    import trimesh

    # An open mesh (not watertight).
    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    cfg = _flat_trim(0.0)
    from kernel.errors import GeometryError

    with pytest.raises(GeometryError):
        apply_trim(open_mesh, cfg)


def test_trim_raises_on_empty_boolean(monkeypatch) -> None:
    """If manifold3d's boolean returns an empty result (e.g., due to a
    pathological cutter that fully contains the helmet), apply_trim must
    raise GeometryError rather than returning an invalid mesh.

    The empty-result branch is defense-in-depth — manifold3d rarely returns
    an empty result on real helmet inputs even when the cutter is large —
    so we trigger it by monkey-patching `_to_manifold` to return a manifold
    whose subtraction yields empty.
    """
    import manifold3d

    import kernel.operations.trim as trim_mod
    from kernel.errors import GeometryError

    head = generate_test_head(HeadConfig())
    shell, _i, _o = build_shell(head, ShellConfig())

    real_to_manifold = trim_mod._to_manifold
    call_count = {"n": 0}

    def fake_to_manifold(mesh):
        call_count["n"] += 1
        m = real_to_manifold(mesh)
        # The second call wraps the cutter — replace it with a giant cube
        # that fully contains the helmet so `helmet - cutter` is empty.
        if call_count["n"] == 2:
            return manifold3d.Manifold.cube([1000.0, 1000.0, 1000.0], center=True)
        return m

    monkeypatch.setattr(trim_mod, "_to_manifold", fake_to_manifold)

    with pytest.raises(GeometryError, match="empty manifold"):
        apply_trim(shell, _flat_trim(0.0))


def test_trim_raises_on_invalid_cutter(monkeypatch) -> None:
    """If the constructed cutting body is not a valid volume, apply_trim must raise."""
    import trimesh

    import kernel.operations.trim as trim_mod
    from kernel.errors import GeometryError

    head = generate_test_head(HeadConfig())
    shell, _i, _o = build_shell(head, ShellConfig())

    # Replace _build_cutting_body with one that returns a degenerate open mesh.
    def fake_build(*_a, **_kw):
        return trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )

    monkeypatch.setattr(trim_mod, "_build_cutting_body", fake_build)

    with pytest.raises(GeometryError, match="not a valid volume"):
        apply_trim(shell, _flat_trim(0.0))


def test_duplicate_azimuths_rejected_by_validator() -> None:
    """The TrimConfig validator rejects duplicate azimuths."""
    with pytest.raises(ValueError, match="distinct azimuths"):
        TrimConfig(
            control_points=[
                TrimControlPoint(azimuth_deg=0.0, elevation_deg=0.0),
                TrimControlPoint(azimuth_deg=0.0, elevation_deg=10.0),
                TrimControlPoint(azimuth_deg=180.0, elevation_deg=0.0),
                TrimControlPoint(azimuth_deg=270.0, elevation_deg=0.0),
            ]
        )
