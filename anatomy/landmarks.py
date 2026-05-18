"""Landmark schema, JSON I/O, and snap-to-surface.

Invariant #8 (CLAUDE.md): landmarks are anchored to mesh geometry, never
free space. `snap_to_surface` returns the nearest on-surface point plus the
face / barycentric anchor so a landmark survives any rigid mesh transform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator

from anatomy.config import REGISTRATION_LANDMARKS, Frame, LandmarkName

logger = logging.getLogger("anatomy")


class Landmark(BaseModel):
    """One annotated cranial landmark.

    `method`:
        manual  — picked by a human in the annotation tool (then snapped)
        snapped — programmatically snapped to a surface from a raw point
        derived — computed analytically from a synthetic-head HeadConfig
    """

    model_config = ConfigDict(frozen=True)

    name: LandmarkName
    position_mm: tuple[float, float, float]
    method: Literal["manual", "snapped", "derived"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""

    def as_array(self) -> np.ndarray:
        return np.asarray(self.position_mm, dtype=np.float64)


class LandmarkSet(BaseModel):
    """A collection of landmarks in a known reference frame.

    `frame` records whether positions are raw-scan ("source") or registered
    ("canonical"). Registration must convert a source set to a canonical set;
    measurements only ever run on a canonical set (invariant #7).
    """

    model_config = ConfigDict(frozen=True)

    frame: Frame
    landmarks: dict[LandmarkName, Landmark]

    @model_validator(mode="after")
    def _key_matches_name(self) -> LandmarkSet:
        for key, lm in self.landmarks.items():
            if key != lm.name:
                raise ValueError(
                    f"landmark dict key {key!r} does not match landmark.name {lm.name!r}"
                )
        return self

    @model_validator(mode="after")
    def _registration_landmarks_present(self) -> LandmarkSet:
        missing = [n for n in REGISTRATION_LANDMARKS if n not in self.landmarks]
        if missing:
            raise ValueError(
                "LandmarkSet missing required registration landmarks: "
                + ", ".join(m.value for m in missing)
            )
        return self

    # -- JSON I/O ---------------------------------------------------------- #

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> LandmarkSet:
        return cls.model_validate_json(text)

    # -- convenience ------------------------------------------------------- #

    def positions(self) -> dict[LandmarkName, np.ndarray]:
        return {name: lm.as_array() for name, lm in self.landmarks.items()}

    def position_tuples(self) -> dict[LandmarkName, tuple[float, float, float]]:
        return {name: lm.position_mm for name, lm in self.landmarks.items()}


@dataclass(frozen=True)
class SnapResult:
    """Result of snapping an arbitrary 3D point onto a mesh surface.

    Either `vertex_id` is set (the snap landed on a mesh vertex) or the point
    lies inside `face_id` at the given barycentric coordinates. `position_mm`
    is always the exact on-surface point.
    """

    position_mm: tuple[float, float, float]
    face_id: int
    barycentric: tuple[float, float, float]
    vertex_id: int | None
    distance_mm: float


def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of `point` (assumed on/near the triangle plane)
    with respect to triangle `tri` (3x3). Deterministic, closed-form."""
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = point - tri[0]
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        # Degenerate triangle — fall back to nearest-vertex weighting.
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def snap_to_surface(point: ArrayLike, mesh: trimesh.Trimesh) -> SnapResult:
    """Snap an arbitrary 3D point to the nearest point on the mesh surface.

    Uses trimesh.proximity.closest_point (deterministic for the pinned
    trimesh version). If the snapped point coincides with a mesh vertex
    (within VERTEX_EPS_MM) the vertex id is recorded; otherwise the
    barycentric anchor on the closest face is recorded.
    """
    pt = np.asarray(point, dtype=np.float64).reshape(1, 3)
    closest, distances, face_ids = trimesh.proximity.closest_point(mesh, pt)
    snapped = np.asarray(closest[0], dtype=np.float64)
    face_id = int(face_ids[0])
    dist = float(distances[0])

    tri = np.asarray(mesh.triangles[face_id], dtype=np.float64)
    bary = _barycentric(snapped, tri)

    vertex_eps_mm = 1e-6
    tri_vert_ids = np.asarray(mesh.faces[face_id], dtype=np.int64)
    vert_dists = np.linalg.norm(tri - snapped, axis=1)
    nearest_corner = int(np.argmin(vert_dists))
    vertex_id: int | None = None
    if vert_dists[nearest_corner] < vertex_eps_mm:
        vertex_id = int(tri_vert_ids[nearest_corner])

    return SnapResult(
        position_mm=(float(snapped[0]), float(snapped[1]), float(snapped[2])),
        face_id=face_id,
        barycentric=(float(bary[0]), float(bary[1]), float(bary[2])),
        vertex_id=vertex_id,
        distance_mm=dist,
    )
