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
    head = generate_test_head(HeadConfig())
    shell, _inner, _outer = build_shell(head, ShellConfig())
    # Flat trim at elevation 0° cuts the helmet at z = bbox_radius * 0 = z_centroid.
    cfg = _flat_trim(0.0)
    trimmed = apply_trim(shell, cfg)
    assert trimmed.is_winding_consistent
    assert trimmed.is_volume
    assert trimmed.is_watertight
    # Trim removed the bottom half, so trimmed.volume should be ~half the shell.
    assert trimmed.volume < shell.volume * 0.7


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
