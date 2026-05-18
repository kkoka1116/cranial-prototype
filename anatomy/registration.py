"""Landmark-based rigid registration into the canonical cranial frame.

Invariant #7 (CLAUDE.md): measurements only ever run on a *canonical* set.
This module converts a ``source``-frame ``LandmarkSet`` (raw scan coordinates)
into a ``canonical``-frame one. The canonical frame is defined by the project
invariant:

    Origin between the ear canals at skull base.
    +X = patient's right, +Y = anterior (toward face), +Z = superior. Right-handed.

The transform is a pure rigid (rotation + translation) map derived
analytically from four landmarks (glabella, vertex, the two tragia). No
scaling, no shear: a registered head keeps every inter-landmark distance, so
downstream metrics (cephalic index, CVAI) are unaffected by registration.

This module makes no decisions and contains no AI; it is the deterministic
geometry side of the system.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import trimesh

from anatomy.config import LandmarkName
from anatomy.errors import RegistrationError
from anatomy.landmarks import Landmark, LandmarkSet

logger = logging.getLogger("anatomy")

# Fraction of the glabella->vertex distance the ear-canal midpoint sits *above*
# the canonical skull-base plane. Dropping the origin by this amount along the
# (provisional) superior axis places the canonical origin at skull base rather
# than at ear-canal height.
TRAGION_SUPRABASAL_FRACTION: float = 0.20

# Below this magnitude a vector is treated as degenerate (colinear / coincident
# landmarks) and registration is refused rather than producing garbage axes.
_MIN_VEC_NORM_MM: float = 1e-9


def _norm(vec: np.ndarray, what: str) -> float:
    """Euclidean norm of ``vec``; raise RegistrationError if ~zero.

    A near-zero norm here always means the input landmarks are coincident or
    colinear, which makes the canonical frame undefined.
    """
    n = float(np.linalg.norm(vec))
    if n < _MIN_VEC_NORM_MM:
        raise RegistrationError(
            f"Degenerate landmark geometry: {what} has near-zero length "
            f"({n:.3e} mm). Landmarks are likely coincident or colinear."
        )
    return n


def compute_canonical_transform(
    landmarks: Mapping[LandmarkName, tuple[float, float, float]],
) -> np.ndarray:
    """Build the 4x4 homogeneous source->canonical rigid transform.

    Maps a source-frame point ``p`` to canonical via
    ``p_canon = (M @ [px, py, pz, 1])[:3]``.

    Requires GLABELLA, VERTEX, TRAGION_LEFT and TRAGION_RIGHT. Note VERTEX is
    required here even though the minimal ``LandmarkSet`` only guarantees the
    four ``REGISTRATION_LANDMARKS`` (glabella, opisthocranion, both tragia):
    registration additionally needs the vertex to fix the superior axis.

    Raises:
        RegistrationError: a required landmark is missing, or the landmark
            geometry is degenerate (coincident/colinear).
    """
    required = (
        LandmarkName.GLABELLA,
        LandmarkName.OPISTHOCRANION,
        LandmarkName.VERTEX,
        LandmarkName.TRAGION_LEFT,
        LandmarkName.TRAGION_RIGHT,
    )
    missing = [n for n in required if n not in landmarks]
    if missing:
        raise RegistrationError(
            "compute_canonical_transform missing required landmark(s): "
            + ", ".join(m.value for m in missing)
            + " (vertex is required for registration in addition to the "
            "minimal registration landmark set)."
        )

    g = np.asarray(landmarks[LandmarkName.GLABELLA], dtype=np.float64)
    o = np.asarray(landmarks[LandmarkName.OPISTHOCRANION], dtype=np.float64)
    v = np.asarray(landmarks[LandmarkName.VERTEX], dtype=np.float64)
    tl = np.asarray(landmarks[LandmarkName.TRAGION_LEFT], dtype=np.float64)
    tr = np.asarray(landmarks[LandmarkName.TRAGION_RIGHT], dtype=np.float64)

    tmid = (tl + tr) / 2.0

    # +Y (anterior) is the *horizontal* AP axis: opisthocranion -> glabella.
    # Using (glabella - origin) instead would tilt the whole frame, because
    # glabella sits at mid-head-height, not on the transverse plane through
    # the ear canals. The opisthocranion->glabella line is the anatomically
    # level AP axis, so the registered transverse plane is truly horizontal
    # (this keeps CVAI / circumference frame-consistent).
    # Axis vectors use ``_ax`` suffixes (x_ax = canonical +X, etc.) to satisfy
    # the project's pep8-naming lint.
    ap_raw = g - o
    y_ax = ap_raw / _norm(ap_raw, "glabella - opisthocranion (AP +Y)")

    # Provisional superior from the ear-canal midpoint, orthogonalized vs +Y.
    z_raw = v - tmid
    z0 = z_raw - (z_raw @ y_ax) * y_ax
    z0 = z0 / _norm(z0, "vertex - tragion_midpoint orthogonalized (provisional +Z)")

    # Drop the origin from ear-canal height down to the skull-base plane.
    gv = _norm(g - v, "glabella - vertex")
    origin = tmid - TRAGION_SUPRABASAL_FRACTION * gv * z0

    # Final basis. +Y is the (origin-independent) horizontal AP direction.
    zt = v - origin
    z_ax = zt - (zt @ y_ax) * y_ax
    z_ax = z_ax / _norm(z_ax, "vertex - origin orthogonalized (final +Z)")

    x_ax = np.cross(y_ax, z_ax)
    x_ax = x_ax / _norm(x_ax, "Y x Z (final +X)")

    # Clean re-orthogonalization so the basis is exactly orthonormal despite
    # floating-point round-off in the cross products above.
    z_ax = np.cross(x_ax, y_ax)
    z_ax = z_ax / _norm(z_ax, "X x Y (re-orthogonalized +Z)")

    # canonical = R @ (p - origin); rows are the canonical world axes.
    rot = np.vstack([x_ax, y_ax, z_ax]).astype(np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ origin
    return transform


def apply_transform(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    """Return a NEW mesh with vertices mapped by the 4x4 ``transform``.

    The input mesh is never mutated. Faces are preserved verbatim and
    ``process=False`` keeps trimesh from silently merging/reordering vertices
    (determinism invariant).
    """
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4):
        raise RegistrationError(f"transform must be 4x4, got shape {t.shape}.")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    homog = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float64)])
    new_verts = (homog @ t.T)[:, :3]

    return trimesh.Trimesh(
        vertices=new_verts,
        faces=np.asarray(mesh.faces).copy(),
        process=False,
    )


def _verify_canonical(canon: LandmarkSet) -> None:
    """Assert the canonical-frame sign conventions hold.

    Glabella must be anterior (+Y), vertex superior (+Z), right tragion on the
    patient's right (+X). A failure almost always means a left/right landmark
    swap upstream.
    """
    pos = canon.positions()
    glabella_y = float(pos[LandmarkName.GLABELLA][1])
    vertex_z = float(pos[LandmarkName.VERTEX][2])
    tragion_right_x = float(pos[LandmarkName.TRAGION_RIGHT][0])

    failures: list[str] = []
    if not glabella_y > 0.0:
        failures.append(f"glabella.y = {glabella_y:.3f} (expected > 0)")
    if not vertex_z > 0.0:
        failures.append(f"vertex.z = {vertex_z:.3f} (expected > 0)")
    if not tragion_right_x > 0.0:
        failures.append(f"tragion_right.x = {tragion_right_x:.3f} (expected > 0)")

    if failures:
        raise RegistrationError(
            "Canonical-frame sign check failed ("
            + "; ".join(failures)
            + "); this is most likely a left/right landmark swap."
        )


def register(landmarks: LandmarkSet) -> tuple[np.ndarray, LandmarkSet]:
    """Register a source-frame ``LandmarkSet`` into the canonical frame.

    Returns the 4x4 source->canonical transform and a NEW ``LandmarkSet``
    (frame ``"canonical"``) with every landmark rigidly transformed. The
    original set is frozen and is never mutated. Method/confidence/notes are
    preserved per landmark; the rigid map does not change how a point was
    derived.

    Raises:
        RegistrationError: input frame is not ``"source"``, a required
            landmark is missing, geometry is degenerate, or the resulting
            frame fails the canonical sign checks.
    """
    if landmarks.frame != "source":
        raise RegistrationError(
            f"register expects a source-frame LandmarkSet, got " f"frame={landmarks.frame!r}."
        )

    positions = landmarks.position_tuples()
    transform = compute_canonical_transform(positions)

    new_landmarks: dict[LandmarkName, Landmark] = {}
    for name, lm in landmarks.landmarks.items():
        p = np.asarray(lm.position_mm, dtype=np.float64)
        p_canon = (transform @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64))[:3]
        new_landmarks[name] = Landmark(
            name=lm.name,
            position_mm=(
                float(p_canon[0]),
                float(p_canon[1]),
                float(p_canon[2]),
            ),
            method=lm.method,
            confidence=lm.confidence,
            notes=lm.notes,
        )

    canon = LandmarkSet(frame="canonical", landmarks=new_landmarks)
    _verify_canonical(canon)

    logger.info(
        "registered landmark set: %d landmarks source->canonical",
        len(new_landmarks),
    )
    return transform, canon
