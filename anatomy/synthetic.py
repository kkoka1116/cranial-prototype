"""Derive a canonical-frame landmark set from a synthetic-head HeadConfig.

This module bridges the deterministic kernel primitive
(`kernel.primitives.test_head.generate_test_head`) and the anatomy landmark
schema. Given the *parameters* of a synthetic head, it computes analytic
target points on the *undeformed* parametric ellipsoid surface, then snaps
each target onto the *real, deformed* generated mesh.

Why snap onto the deformed mesh? The analytic targets describe where each
landmark sits on the clean parametric ellipsoid. Snapping pins them onto the
actual generated surface so that occipital flattening, frontal bossing, and
brachycephaly compression measurably move the landmarks — which is exactly
what downstream measurements (week 2) need to detect asymmetry.

Determinism: `generate_test_head` and `trimesh.proximity.closest_point`
(used by `snap_to_surface`) are deterministic for the pinned library
versions, so `derive_landmarks_from_synthetic` is bit-for-bit reproducible.

Reference frame (CLAUDE.md invariant #4): origin between the ear canals at
skull base; +X patient's right; +Y anterior; +Z superior. The synthetic
head spans z in [0, height_mm].

Parametric mapping mirrored from `generate_test_head`: a base unit-sphere
direction ``u = (ux, uy, uz)`` maps to the undeformed surface point::

    P = (a * ux,  b * uy * bf,  c * uz + H / 2)

where ``a = width_mm / 2``, ``b = length_mm / 2``, ``c = height_mm / 2``,
``H = height_mm`` and ``bf = brachycephaly_factor``.
"""

from __future__ import annotations

from math import cos, radians, sin, sqrt

import numpy as np

from anatomy.config import LandmarkName
from anatomy.landmarks import Landmark, LandmarkSet, snap_to_surface
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head

# Fraction of the glabella->vertex span used to place the tragion above the
# skull base. Tragion sits low on the side of the head, at ear-canal level.
TRAGION_SUPRABASAL_FRACTION: float = 0.20


def _map_point(
    direction: tuple[float, float, float] | np.ndarray,
    a_axis: float,
    b_axis: float,
    c_axis: float,
    height_mm: float,
    brachy_factor: float,
) -> tuple[float, float, float]:
    """Map a unit-sphere direction to the undeformed parametric surface point.

    Implements ``P = (a*ux, b*uy*bf, c*uz + H/2)`` — the inverse of the
    forward construction in ``generate_test_head`` (anisotropic scale, then
    Y-compression, then +Z lift).
    """
    ux, uy, uz = float(direction[0]), float(direction[1]), float(direction[2])
    return (
        a_axis * ux,
        b_axis * uy * brachy_factor,
        c_axis * uz + height_mm / 2.0,
    )


def _unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3-vector. Used for spherical-coordinate target directions."""
    arr = np.asarray(vec, dtype=np.float64)
    length = float(np.linalg.norm(arr))
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    arr = arr / length
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _analytic_targets(
    head_params: HeadConfig,
) -> dict[LandmarkName, tuple[float, float, float]]:
    """Compute the 12 analytic target points on the undeformed surface.

    All targets are in the canonical frame. They are subsequently snapped
    onto the real deformed mesh by the caller.
    """
    width_mm = head_params.width_mm
    length_mm = head_params.length_mm
    height_mm = head_params.height_mm
    brachy_factor = head_params.brachycephaly_factor

    a_axis = width_mm / 2.0
    b_axis = length_mm / 2.0
    c_axis = height_mm / 2.0

    def mapped(direction: tuple[float, float, float]) -> tuple[float, float, float]:
        return _map_point(direction, a_axis, b_axis, c_axis, height_mm, brachy_factor)

    targets: dict[LandmarkName, tuple[float, float, float]] = {}

    # Midline / axial landmarks.
    targets[LandmarkName.VERTEX] = mapped((0.0, 0.0, 1.0))
    targets[LandmarkName.GLABELLA] = mapped((0.0, 1.0, 0.0))
    targets[LandmarkName.OPISTHOCRANION] = mapped((0.0, -1.0, 0.0))

    # Nasion: slightly below glabella on the anterior midline.
    nasion_theta = radians(15.0)
    targets[LandmarkName.NASION] = mapped((0.0, cos(nasion_theta), -sin(nasion_theta)))

    # Euryon: maximum width, lateral. +X is patient's right.
    targets[LandmarkName.EURYON_RIGHT] = mapped((1.0, 0.0, 0.0))
    targets[LandmarkName.EURYON_LEFT] = mapped((-1.0, 0.0, 0.0))

    # Tragion: AP-centered (y == 0), supra-basal at a fixed fraction of the
    # glabella->vertex span. Solve for uz on the ellipsoid at uy = 0.
    glabella = targets[LandmarkName.GLABELLA]
    vertex = targets[LandmarkName.VERTEX]
    gv = sqrt(
        (glabella[0] - vertex[0]) ** 2
        + (glabella[1] - vertex[1]) ** 2
        + (glabella[2] - vertex[2]) ** 2
    )
    z_t = TRAGION_SUPRABASAL_FRACTION * gv
    uz_t = (z_t - height_mm / 2.0) / c_axis
    uz_t = max(-1.0, min(1.0, uz_t))
    ux_t = sqrt(max(0.0, 1.0 - uz_t * uz_t))
    targets[LandmarkName.TRAGION_RIGHT] = (a_axis * ux_t, 0.0, z_t)
    targets[LandmarkName.TRAGION_LEFT] = (-a_axis * ux_t, 0.0, z_t)

    # Frontotemporale: anterior-lateral junction.
    ft_az = radians(55.0)
    ft_el = radians(10.0)
    ft_dir = _unit(
        (
            sin(ft_az) * cos(ft_el),
            cos(ft_az) * cos(ft_el),
            sin(ft_el),
        )
    )
    ft_right = mapped(ft_dir)
    targets[LandmarkName.FRONTOTEMPORALE_RIGHT] = ft_right
    targets[LandmarkName.FRONTOTEMPORALE_LEFT] = (
        -ft_right[0],
        ft_right[1],
        ft_right[2],
    )

    # Parietal eminence: posterior-lateral prominence.
    pe_az = radians(70.0)
    pe_el = radians(40.0)
    pe_dir = _unit(
        (
            sin(pe_az) * cos(pe_el),
            -0.30 * cos(pe_el),
            sin(pe_el),
        )
    )
    pe_right = mapped(pe_dir)
    targets[LandmarkName.PARIETAL_EMINENCE_RIGHT] = pe_right
    targets[LandmarkName.PARIETAL_EMINENCE_LEFT] = (
        -pe_right[0],
        pe_right[1],
        pe_right[2],
    )

    return targets


def derive_landmarks_from_synthetic(head_params: HeadConfig) -> LandmarkSet:
    """Derive a canonical-frame ``LandmarkSet`` from synthetic-head params.

    Steps:

    1. Generate the real, deformed synthetic-head mesh.
    2. Compute analytic target points for all 12 landmarks on the undeformed
       parametric surface.
    3. Snap every target onto the real mesh surface (pins landmarks onto the
       deformed geometry so downstream measurements see the deformations).
    4. Return a frozen ``LandmarkSet`` (frame="canonical") with every
       landmark ``method="derived"``, ``confidence=1.0``.

    Deterministic for the pinned library versions.
    """
    mesh = generate_test_head(head_params)
    targets = _analytic_targets(head_params)

    landmarks: dict[LandmarkName, Landmark] = {}
    # Iterate in the enum's fixed definition order for deterministic build.
    for name in LandmarkName:
        target = targets[name]
        snapped = snap_to_surface(target, mesh)
        landmarks[name] = Landmark(
            name=name,
            position_mm=snapped.position_mm,
            method="derived",
            confidence=1.0,
            notes="derived from synthetic HeadConfig; snapped to generated mesh",
        )

    return LandmarkSet(frame="canonical", landmarks=landmarks)
