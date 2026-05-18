"""Tests for the landmark schema, JSON round-trip, and snap-to-surface."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from anatomy.config import REGISTRATION_LANDMARKS, LandmarkName
from anatomy.landmarks import Landmark, LandmarkSet, snap_to_surface


def _full_set(frame: str = "canonical") -> LandmarkSet:
    lms = {}
    for i, name in enumerate(LandmarkName):
        lms[name] = Landmark(
            name=name,
            position_mm=(float(i), float(i) + 1.0, float(i) + 2.0),
            method="derived",
        )
    return LandmarkSet(frame=frame, landmarks=lms)


def _min_set() -> LandmarkSet:
    lms = {
        n: Landmark(name=n, position_mm=(1.0, 2.0, 3.0), method="manual")
        for n in REGISTRATION_LANDMARKS
    }
    return LandmarkSet(frame="source", landmarks=lms)


def test_json_round_trip_full_set() -> None:
    s = _full_set()
    restored = LandmarkSet.from_json(s.to_json())
    assert restored == s
    assert restored.frame == "canonical"
    assert len(restored.landmarks) == 12


def test_min_registration_set_is_valid() -> None:
    s = _min_set()
    assert set(s.landmarks) == set(REGISTRATION_LANDMARKS)


def test_incomplete_set_rejected() -> None:
    with pytest.raises(ValidationError):
        LandmarkSet(
            frame="source",
            landmarks={
                LandmarkName.GLABELLA: Landmark(
                    name=LandmarkName.GLABELLA,
                    position_mm=(0.0, 0.0, 0.0),
                    method="manual",
                )
            },
        )


def test_key_name_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        LandmarkSet(
            frame="source",
            landmarks={
                LandmarkName.GLABELLA: Landmark(
                    name=LandmarkName.NASION,
                    position_mm=(0.0, 0.0, 0.0),
                    method="manual",
                )
            },
        )


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        Landmark(
            name=LandmarkName.VERTEX,
            position_mm=(0.0, 0.0, 0.0),
            method="derived",
            confidence=1.5,
        )


def test_snap_to_surface_on_known_box() -> None:
    """A point well above a box face snaps straight down to that face."""
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))  # centered at origin
    res = snap_to_surface([0.0, 0.0, 50.0], box)
    # Closest point is the top face at z = +5.
    assert res.position_mm[2] == pytest.approx(5.0, abs=1e-6)
    assert res.distance_mm == pytest.approx(45.0, abs=1e-6)
    # Barycentric coords are a partition of unity.
    assert sum(res.barycentric) == pytest.approx(1.0, abs=1e-9)
    assert 0 <= res.face_id < len(box.faces)


def test_snap_to_surface_on_vertex_records_vertex_id() -> None:
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    corner = box.vertices[0]
    # Push outward along the corner direction so the closest point is exactly
    # that vertex.
    probe = corner + corner / np.linalg.norm(corner) * 10.0
    res = snap_to_surface(probe, box)
    assert res.vertex_id is not None
    np.testing.assert_allclose(res.position_mm, corner, atol=1e-6)


def test_positions_helper_returns_arrays() -> None:
    s = _full_set()
    pos = s.positions()
    assert set(pos) == set(LandmarkName)
    assert all(isinstance(v, np.ndarray) and v.shape == (3,) for v in pos.values())
