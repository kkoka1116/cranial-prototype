"""Tests for landmark-based registration into the canonical cranial frame.

The synthetic head is the Week-1 ellipsoid: semi-axes a=W/2 (X), b=L/2 (Y),
c=H/2 (Z), centered in XY at the origin, z spanning [0, H] (ellipsoid centre
at (0, 0, H/2)). Canonical landmark positions are derived analytically so we
can apply a known rigid transform, register back, and assert recovery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from anatomy.config import LandmarkName
from anatomy.errors import RegistrationError
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.registration import (
    TRAGION_SUPRABASAL_FRACTION,
    apply_transform,
    compute_canonical_transform,
    register,
)

# Synthetic-head defaults (mm).
_W = 130.0
_L = 165.0
_H = 115.0


def _canonical_landmark_points() -> dict[LandmarkName, np.ndarray]:
    """Analytic canonical positions of all twelve landmarks on the ellipsoid.

    The four registration-critical points (glabella, vertex, both tragia) are
    placed exactly per the spec. The remaining eight get deterministic,
    roughly on-surface placeholders so the ``LandmarkSet`` validator is happy;
    they do not influence the computed transform.
    """
    a = _W / 2.0
    b = _L / 2.0
    c = _H / 2.0

    glabella = np.array([0.0, b, c], dtype=np.float64)
    vertex = np.array([0.0, 0.0, _H], dtype=np.float64)
    opisthocranion = np.array([0.0, -b, c], dtype=np.float64)

    gv = math.sqrt(b**2 + c**2)
    z_t = TRAGION_SUPRABASAL_FRACTION * gv
    uz = (z_t - c) / c
    ux = math.sqrt(max(0.0, 1.0 - uz**2))
    x_t = a * ux
    tragion_right = np.array([+x_t, 0.0, z_t], dtype=np.float64)
    tragion_left = np.array([-x_t, 0.0, z_t], dtype=np.float64)

    pts: dict[LandmarkName, np.ndarray] = {
        LandmarkName.GLABELLA: glabella,
        LandmarkName.VERTEX: vertex,
        LandmarkName.OPISTHOCRANION: opisthocranion,
        LandmarkName.TRAGION_LEFT: tragion_left,
        LandmarkName.TRAGION_RIGHT: tragion_right,
        # Deterministic placeholders for the other seven landmarks.
        LandmarkName.NASION: np.array([0.0, b * 0.95, c * 0.7], dtype=np.float64),
        LandmarkName.EURYON_LEFT: np.array([-a, 0.0, c], dtype=np.float64),
        LandmarkName.EURYON_RIGHT: np.array([+a, 0.0, c], dtype=np.float64),
        LandmarkName.FRONTOTEMPORALE_LEFT: np.array([-a * 0.7, b * 0.5, c * 0.9], dtype=np.float64),
        LandmarkName.FRONTOTEMPORALE_RIGHT: np.array(
            [+a * 0.7, b * 0.5, c * 0.9], dtype=np.float64
        ),
        LandmarkName.PARIETAL_EMINENCE_LEFT: np.array(
            [-a * 0.8, -b * 0.3, _H * 0.9], dtype=np.float64
        ),
        LandmarkName.PARIETAL_EMINENCE_RIGHT: np.array(
            [+a * 0.8, -b * 0.3, _H * 0.9], dtype=np.float64
        ),
    }
    return pts


def _rigid_transform(angle_deg: float, axis: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """4x4 rigid transform: rotate ``angle_deg`` about ``axis``, then translate."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = math.radians(angle_deg)
    kx, ky, kz = axis
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    rot = np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _apply4(mat: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (mat @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64))[:3]


def _landmark_set(points: dict[LandmarkName, np.ndarray], frame: str = "source") -> LandmarkSet:
    lms = {
        name: Landmark(
            name=name,
            position_mm=(float(p[0]), float(p[1]), float(p[2])),
            method="derived",
        )
        for name, p in points.items()
    }
    return LandmarkSet(frame=frame, landmarks=lms)


def test_register_recovers_known_rigid_transform() -> None:
    """Registering T-displaced landmarks recovers the original canonical
    positions and orientation to sub-0.5mm / sub-0.5deg precision."""
    canonical = _canonical_landmark_points()
    t_known = _rigid_transform(
        23.7,
        np.array([0.3, -0.8, 0.5]),
        np.array([12.0, -7.0, 40.0]),
    )

    source = {name: _apply4(t_known, p) for name, p in canonical.items()}

    m_ref = compute_canonical_transform({n: tuple(p) for n, p in canonical.items()})
    m_src = compute_canonical_transform({n: tuple(p) for n, p in source.items()})

    # Applying m_src to a t_known-displaced point must equal m_ref on the original.
    max_pos_err = 0.0
    for p_canon in canonical.values():
        via_src = _apply4(m_src, _apply4(t_known, p_canon))
        via_ref = _apply4(m_ref, p_canon)
        max_pos_err = max(max_pos_err, float(np.linalg.norm(via_src - via_ref)))
    assert max_pos_err < 0.5, f"position recovery error {max_pos_err} mm"

    # Rotation of (m_src @ t_known) must match rotation of m_ref.
    rot_recovered = (m_src @ t_known)[:3, :3]
    rot_expected = m_ref[:3, :3]
    rot_residual = rot_recovered @ rot_expected.T
    cos_angle = (np.trace(rot_residual) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_err_deg = math.degrees(math.acos(cos_angle))
    assert angle_err_deg < 0.5, f"rotation recovery error {angle_err_deg} deg"

    # Surface the observed errors for the run report.
    print(f"max position recovery error = {max_pos_err:.3e} mm")
    print(f"rotation recovery error     = {angle_err_deg:.3e} deg")


def test_register_is_consistent_idempotentish() -> None:
    """An already-canonical synthetic set re-registers to a valid canonical
    frame and keeps the glabella<->opisthocranion distance (rigid invariance)."""
    canonical = _canonical_landmark_points()
    src = _landmark_set(canonical, frame="source")

    g0 = canonical[LandmarkName.GLABELLA]
    o0 = canonical[LandmarkName.OPISTHOCRANION]
    dist_before = float(np.linalg.norm(g0 - o0))

    _, canon = register(src)
    pos = canon.positions()

    # _verify_canonical sign conditions hold.
    assert pos[LandmarkName.GLABELLA][1] > 0.0
    assert pos[LandmarkName.VERTEX][2] > 0.0
    assert pos[LandmarkName.TRAGION_RIGHT][0] > 0.0

    dist_after = float(
        np.linalg.norm(pos[LandmarkName.GLABELLA] - pos[LandmarkName.OPISTHOCRANION])
    )
    assert dist_after == pytest.approx(dist_before, abs=1e-6)


def test_left_right_swap_raises() -> None:
    """Swapping the tragia puts tragion_right at negative canonical X, which
    the sign check must reject."""
    canonical = _canonical_landmark_points()
    swapped = dict(canonical)
    swapped[LandmarkName.TRAGION_LEFT] = canonical[LandmarkName.TRAGION_RIGHT]
    swapped[LandmarkName.TRAGION_RIGHT] = canonical[LandmarkName.TRAGION_LEFT]
    src = _landmark_set(swapped, frame="source")

    with pytest.raises(RegistrationError, match="left/right"):
        register(src)


def test_missing_vertex_raises() -> None:
    """compute_canonical_transform needs VERTEX even though the minimal
    LandmarkSet does not include it."""
    canonical = _canonical_landmark_points()
    no_vertex = {n: tuple(p) for n, p in canonical.items() if n != LandmarkName.VERTEX}
    with pytest.raises(RegistrationError, match="vertex"):
        compute_canonical_transform(no_vertex)


def test_apply_transform_does_not_mutate_input() -> None:
    """apply_transform returns a fresh mesh and leaves the input untouched."""
    sphere = trimesh.creation.icosphere(subdivisions=2)
    before = np.array(sphere.vertices, dtype=np.float64, copy=True)

    t_known = _rigid_transform(0.0, np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    out = apply_transform(sphere, t_known)

    assert out is not sphere
    np.testing.assert_array_equal(sphere.vertices, before)
    np.testing.assert_allclose(out.vertices, before + np.array([1.0, 2.0, 3.0]), atol=1e-9)
    np.testing.assert_array_equal(out.faces, sphere.faces)


def test_register_requires_source_frame() -> None:
    """register refuses a set that is already in the canonical frame."""
    canonical = _canonical_landmark_points()
    canon_set = _landmark_set(canonical, frame="canonical")
    with pytest.raises(RegistrationError, match="source"):
        register(canon_set)
