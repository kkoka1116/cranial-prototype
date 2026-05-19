"""Strap-mount tab fusion on a finished helmet shell.

A *strap mount* is a small solid tab fused to the outside of the helmet near
the lower trim edge. A clinician threads a chin/retention strap through it, so
it must be a robust, knife-edge-free protrusion that is structurally part of
the shell (a single connected solid, not a glued-on appendage).

This is an *additive* Week-4-permitted kernel primitive: it takes a finished,
watertight helmet shell (e.g. from
:func:`kernel.operations.shell.build_shell` followed by
:func:`kernel.operations.trim.apply_trim`) and returns a new watertight shell.
It never mutates its input and contains zero AI / no decisions —
caller-supplied parameters in, deterministic geometry out.

Tab construction
----------------
The tab is built as ONE closed, watertight lofted prism:

  - A **rounded reinforcement base disk** of radius ``reinforcement_radius_mm``
    lying in the shell's local tangent plane at ``anchor_mm``. It is recessed
    *inward* (toward the shell centroid) by a fixed depth so it bites well
    into the wall — this is what guarantees the union is a single connected
    solid with a smooth, filleted root rather than a knife edge.
  - A **rectangular strap cross-section** of width ``strap_width_mm`` at the
    far (outboard) end of the tab.
  - The two profiles are lofted together with a smoothly interpolated
    cross-section (disk → rounded-rectangle → rectangle) so the wall of the
    tab is continuous and the base meets the shell with a rounded fillet of
    radius ``reinforcement_radius_mm``.

The loft axis is the local outward direction (unit vector from the shell
centroid toward ``anchor_mm`` — an acceptable surface-normal proxy per spec)
tilted by ``angle_deg`` toward ``-Z`` (inferior), so the strap exits the
helmet pointing down-and-out the way a real chin strap runs.

How the single connected solid is guaranteed
---------------------------------------------
The base profile sits a fixed distance *behind* ``anchor_mm`` along the
inward axis (``_BASE_RECESS_MM``), which comfortably exceeds any realistic
shell wall thickness. The tab therefore fully spans and overlaps the wall, so
the manifold3d boolean *union* with the helmet produces one connected
component. We assert that explicitly (``split`` count == 1) and raise
``GeometryError`` rather than ever returning a floating tab (invariant #6).

Sign of the volume change
-------------------------
A boolean *union* only ever adds material, so the strapped shell's
``volume`` (solid wall material) is *strictly larger* than the input's, and a
larger ``strap_width_mm`` or ``reinforcement_radius_mm`` makes a bigger tab,
so the volume increases monotonically with either.

Determinism
-----------
The loft tessellation is fixed (``_LOFT_STATIONS`` × ``_SECTORS``) so the tab
topology is identical across runs. After the manifold3d boolean we convert
back via :func:`kernel.operations.shell._from_manifold`, which canonicalizes
the vertex / face ordering (lexsort + face rotation/sort). We never call
``trimesh.fix_normals`` on the boolean output (per the repo determinism
note); ``fix_normals`` *is* used on the tab we construct ourselves, which is
empirically deterministic for the pinned trimesh version. Identical arguments
therefore produce a byte-identical STL export.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from kernel.errors import GeometryError
from kernel.operations.shell import _from_manifold, _to_manifold

logger = logging.getLogger(__name__)

# Loft tessellation. Fixed so the tab topology — and therefore the boolean
# output — is identical across runs (determinism, invariant #2).
#
# ``_SECTORS`` points trace each cross-sectional ring; ``_LOFT_STATIONS``
# rings are stacked from the recessed base to the outboard tip.
_SECTORS = 64
_LOFT_STATIONS = 16

# How far (mm) the tab's base profile sits *behind* ``anchor_mm`` along the
# inward axis, i.e. recessed into the shell wall. Must comfortably exceed any
# realistic helmet wall thickness so the tab fully overlaps the wall and the
# union is a single connected solid (not a tangent kiss).
_BASE_RECESS_MM = 14.0

# How far (mm) the tab protrudes outboard from ``anchor_mm`` along the loft
# axis. The strap slot lives near this outboard end.
_PROTRUSION_MM = 18.0


def _local_outward_frame(
    shell_mesh: trimesh.Trimesh, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an orthonormal frame ``(u, v, axis)`` at the anchor.

    ``axis`` is the unit vector from the shell centroid to ``anchor`` (the
    local "outward", head-away direction — an acceptable surface-normal proxy
    per spec). ``u`` and ``v`` span the local tangent plane. The frame is
    derived deterministically from the geometry with a fixed tie-breaking
    reference vector (mirrors
    ``kernel.primitives.relief._local_outward_axis``).
    """
    axis = anchor - np.asarray(shell_mesh.centroid, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise GeometryError(
            "strap anchor coincides with the shell centroid; " "cannot derive an outward axis",
            context={"anchor_mm": anchor.tolist()},
        )
    axis = axis / norm
    # Fixed tie-breaking reference so the in-plane frame is reproducible.
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v, axis


def _tilt_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Tilt ``axis`` by ``angle_deg`` toward ``-Z`` (inferior).

    The strap should exit the helmet pointing down-and-out (the way a chin
    strap runs), so we rotate the outward axis within the plane spanned by
    ``axis`` and the projection of ``-Z`` perpendicular to ``axis``. If
    ``axis`` is (anti)parallel to ``Z`` the down-direction is degenerate, so
    we fall back to a fixed in-plane reference for reproducibility.
    """
    down = np.array([0.0, 0.0, -1.0])
    # Component of -Z perpendicular to axis = the in-plane "down" direction.
    perp = down - np.dot(down, axis) * axis
    n = float(np.linalg.norm(perp))
    if n < 1e-9:
        # axis is along ±Z; pick a fixed perpendicular so the result is
        # deterministic (no anatomical "down" exists for a pole anchor).
        perp = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), axis) * axis
        n = float(np.linalg.norm(perp))
    perp = perp / n
    theta = np.deg2rad(angle_deg)
    tilted = np.cos(theta) * axis + np.sin(theta) * perp
    return tilted / float(np.linalg.norm(tilted))


def _profile_ring(
    half_width_mm: float,
    radius_mm: float,
    blend: float,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """One closed cross-sectional ring of ``_SECTORS`` points in the (u, v) plane.

    ``blend`` in ``[0, 1]`` morphs the shape continuously:

      - ``blend == 0``  → a circle of radius ``radius_mm`` (the rounded
        reinforcement base).
      - ``blend == 1``  → a rounded rectangle of half-width ``half_width_mm``
        (the strap cross-section), corners filleted by the residual radius so
        the loft wall never develops a knife edge.

    The same ``_SECTORS`` angular samples are used for every station, so all
    rings have matching topology and the loft side-wall is a clean quad strip.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, _SECTORS, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Circle component (base).
    circ = np.column_stack([radius_mm * cos_a, radius_mm * sin_a])

    # Rounded-rectangle component (strap end): a square-ish slot of half-width
    # ``half_width_mm`` with corners rounded by ``corner_r`` so there is no
    # sharp edge anywhere on the lofted wall.
    corner_r = min(radius_mm, half_width_mm) * 0.5
    flat = half_width_mm - corner_r
    rect_x = np.clip(corner_r * cos_a, -1e18, 1e18) + np.sign(cos_a) * flat
    rect_y = np.clip(corner_r * sin_a, -1e18, 1e18) + np.sign(sin_a) * flat
    rect = np.column_stack([rect_x, rect_y])

    mixed = (1.0 - blend) * circ + blend * rect
    return mixed[:, 0:1] * u + mixed[:, 1:2] * v


def _build_strap_tab(
    anchor: np.ndarray,
    strap_width_mm: float,
    reinforcement_radius_mm: float,
    u: np.ndarray,
    v: np.ndarray,
    inward_axis: np.ndarray,
    loft_axis: np.ndarray,
) -> trimesh.Trimesh:
    """Construct the closed, watertight lofted strap tab.

    The tab has two phases along its length, so that "bite into the wall" is
    decoupled from "strap exit angle":

      - **Root phase** — the base profile is recessed ``_BASE_RECESS_MM``
        straight *inward* along ``inward_axis`` (the untilted local outward
        axis reversed, i.e. toward the shell centroid). This guarantees the
        tab fully overlaps the wall regardless of ``angle_deg``, so the
        boolean union is a single connected solid.
      - **Protrusion phase** — from ``anchor`` the loft sweeps outboard along
        the (possibly tilted) ``loft_axis`` for ``_PROTRUSION_MM``, giving the
        strap its exit direction.

    Topology (deterministic; ``S = _LOFT_STATIONS``, ``N = _SECTORS``):

      - ``S`` cross-sectional rings of ``N`` vertices stacked along the path
        described above, from the recessed base to the outboard tip.
      - the cross-section morphs base-circle → strap-rectangle and the
        bounding size grows from the reinforcement radius to the strap
        half-width, so the root flares into a rounded fillet of radius
        ``reinforcement_radius_mm``.
      - two apex vertices (base centre, tip centre) close the ends with
        triangle fans — the solid is fully closed and manifold.
    """
    half_width_mm = strap_width_mm / 2.0

    # Parametric station positions s in [0, 1] along the tab.
    s = np.linspace(0.0, 1.0, _LOFT_STATIONS)
    # Smoothstep blend so the cross-section transitions C1-continuously from
    # the round base to the rectangular strap end (no kink in the wall).
    blend = s * s * (3.0 - 2.0 * s)
    # The in-plane size eases from the reinforcement radius (root) up to the
    # strap half-width (tip) with the same smoothstep, producing the rounded
    # fillet where the tab meets the shell.
    size = (1.0 - blend) * reinforcement_radius_mm + blend * half_width_mm

    # Centre-line: recessed base (inward) -> anchor -> tilted protrusion.
    # ``frac`` is the fraction of stations in the root phase; the remainder
    # are in the protrusion phase. ``anchor`` sits exactly at the boundary.
    base_pt = anchor - inward_axis * _BASE_RECESS_MM
    tip_pt = anchor + loft_axis * _PROTRUSION_MM
    frac = _BASE_RECESS_MM / (_BASE_RECESS_MM + _PROTRUSION_MM)
    centres = np.empty((_LOFT_STATIONS, 3), dtype=np.float64)
    for k in range(_LOFT_STATIONS):
        if s[k] <= frac:
            t = s[k] / frac if frac > 0.0 else 1.0
            centres[k] = (1.0 - t) * base_pt + t * anchor
        else:
            t = (s[k] - frac) / (1.0 - frac) if frac < 1.0 else 1.0
            centres[k] = (1.0 - t) * anchor + t * tip_pt

    rings: list[np.ndarray] = []
    for k in range(_LOFT_STATIONS):
        ring = _profile_ring(
            half_width_mm=size[k],
            radius_mm=size[k],
            blend=float(blend[k]),
            u=u,
            v=v,
        )
        rings.append(centres[k] + ring)

    n = _SECTORS
    all_ring_verts = np.vstack(rings)
    base_apex = centres[0]
    tip_apex = centres[-1]
    verts = np.vstack([all_ring_verts, base_apex[None, :], tip_apex[None, :]])
    base_apex_idx = _LOFT_STATIONS * n
    tip_apex_idx = _LOFT_STATIONS * n + 1

    def vid(station: int, sec: int) -> int:
        return station * n + (sec % n)

    faces: list[list[int]] = []

    # Base cap fan (station 0). Winding chosen so the cap normal points
    # *inward* (along -loft_axis), opposite the tip cap.
    for sec in range(n):
        faces.append([base_apex_idx, vid(0, sec + 1), vid(0, sec)])

    # Side wall: quad strip between successive stations.
    for st in range(_LOFT_STATIONS - 1):
        for sec in range(n):
            a = vid(st, sec)
            b = vid(st, sec + 1)
            c = vid(st + 1, sec)
            d = vid(st + 1, sec + 1)
            faces.append([a, c, d])
            faces.append([a, d, b])

    # Tip cap fan (last station), opposite winding to the base.
    last = _LOFT_STATIONS - 1
    for sec in range(n):
        faces.append([tip_apex_idx, vid(last, sec), vid(last, sec + 1)])

    tab = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    # fix_normals is empirically deterministic for the pinned trimesh version
    # on a mesh we construct ourselves (same rationale as
    # kernel/operations/trim.py and kernel/primitives/relief.py). It is NOT
    # applied to the boolean output.
    tab.fix_normals()
    return tab


def _validate_args(
    helmet_mesh: trimesh.Trimesh,
    strap_width_mm: float,
    angle_deg: float,
    reinforcement_radius_mm: float,
) -> None:
    """Reject malformed inputs before touching geometry (invariant #6)."""
    if not helmet_mesh.is_volume:
        raise GeometryError(
            "helmet input to apply_strap_mount is not a valid volume",
            context={
                "is_winding_consistent": bool(helmet_mesh.is_winding_consistent),
                "is_watertight": bool(helmet_mesh.is_watertight),
            },
        )
    if strap_width_mm <= 0.0:
        raise GeometryError(f"strap_width_mm must be > 0, got {strap_width_mm}")
    if reinforcement_radius_mm <= 0.0:
        raise GeometryError(f"reinforcement_radius_mm must be > 0, got {reinforcement_radius_mm}")
    if not (-90.0 < angle_deg < 90.0):
        raise GeometryError(f"angle_deg must be in (-90, 90), got {angle_deg}")


def apply_strap_mount(
    helmet_mesh: trimesh.Trimesh,
    anchor_mm: tuple[float, float, float],
    strap_width_mm: float = 20.0,
    angle_deg: float = 35.0,
    reinforcement_radius_mm: float = 8.0,
) -> trimesh.Trimesh:
    """Fuse a strap-attachment tab onto a finished helmet shell.

    Builds one closed, watertight lofted tab (a rounded reinforcement base
    disk lofted out to a rectangular strap cross-section), positions it so it
    overlaps the shell wall at ``anchor_mm``, and boolean-*unions* it onto the
    helmet. The result is a single connected solid with a smooth filleted
    root — never a floating appendage.

    Args:
        helmet_mesh: A watertight, winding-consistent helmet shell in the
            canonical reference frame (origin between the ear canals, +X
            patient right, +Y anterior, +Z superior). Not mutated.
        anchor_mm: ``(x, y, z)`` of the tab root, in mm, in the canonical
            frame. Should lie on / near the helmet's lower trim edge — the
            tab is grown from the local surface there.
        strap_width_mm: Width of the strap cross-section, in mm. Must be > 0.
            Larger ⇒ a bigger tab ⇒ strictly more added material.
        angle_deg: Tilt of the tab away from the local outward normal, toward
            ``-Z`` (inferior), in degrees. Must be in ``(-90, 90)``. Changes
            the tab orientation (the exported mesh differs) without changing
            whether it is fused.
        reinforcement_radius_mm: Radius of the rounded reinforcement base
            where the tab meets the shell, in mm. Must be > 0. Larger ⇒ a
            beefier root ⇒ strictly more added material and a gentler fillet.

    Returns:
        A new ``trimesh.Trimesh`` — the strapped shell, watertight and
        winding-consistent. Its ``volume`` (solid wall material) is strictly
        larger than the input's because the union only adds material.

    Raises:
        kernel.errors.GeometryError: if the input is not a valid volume, the
            arguments are out of range, the boolean produced empty geometry,
            the tab failed to fuse (more than one connected component), or
            the result is not watertight / winding-consistent.
    """
    _validate_args(helmet_mesh, strap_width_mm, angle_deg, reinforcement_radius_mm)

    anchor = np.array(anchor_mm, dtype=np.float64)
    u, v, axis = _local_outward_frame(helmet_mesh, anchor)
    loft_axis = _tilt_axis(axis, angle_deg)

    tab = _build_strap_tab(
        anchor=anchor,
        strap_width_mm=strap_width_mm,
        reinforcement_radius_mm=reinforcement_radius_mm,
        u=u,
        v=v,
        inward_axis=axis,
        loft_axis=loft_axis,
    )
    if not tab.is_volume:
        raise GeometryError(
            "constructed strap tab is not a valid volume",
            context={
                "anchor_mm": anchor.tolist(),
                "strap_width_mm": float(strap_width_mm),
                "reinforcement_radius_mm": float(reinforcement_radius_mm),
                "is_watertight": bool(tab.is_watertight),
                "is_winding_consistent": bool(tab.is_winding_consistent),
            },
        )

    helmet_m = _to_manifold(helmet_mesh)
    tab_m = _to_manifold(tab)
    strapped_m = helmet_m + tab_m
    if strapped_m.is_empty():
        raise GeometryError(
            "strap boolean union produced an empty manifold",
            context={
                "anchor_mm": anchor.tolist(),
                "strap_width_mm": float(strap_width_mm),
                "reinforcement_radius_mm": float(reinforcement_radius_mm),
            },
        )

    strapped = _from_manifold(strapped_m)

    # Fusion gate: the tab must overlap the shell so the union is ONE solid.
    # A disconnected result means the anchor was off the wall and the tab is
    # floating — never return that (invariant #6).
    n_components = len(strapped.split(only_watertight=False))
    if n_components != 1:
        raise GeometryError(
            "strap tab did not fuse to the shell (disconnected result); "
            f"anchor {anchor.tolist()} may be off the wall",
            context={
                "anchor_mm": anchor.tolist(),
                "strap_width_mm": float(strap_width_mm),
                "angle_deg": float(angle_deg),
                "reinforcement_radius_mm": float(reinforcement_radius_mm),
                "n_components": int(n_components),
            },
        )

    # Final invariant gate: never let broken geometry escape the kernel
    # (invariant #6).
    if not strapped.is_winding_consistent or not strapped.is_watertight:
        raise GeometryError(
            "strap boolean produced non-watertight / non-winding-consistent " "geometry",
            context={
                "anchor_mm": anchor.tolist(),
                "strap_width_mm": float(strap_width_mm),
                "angle_deg": float(angle_deg),
                "reinforcement_radius_mm": float(reinforcement_radius_mm),
                "is_watertight": bool(strapped.is_watertight),
                "is_winding_consistent": bool(strapped.is_winding_consistent),
            },
        )

    logger.debug(
        "apply_strap_mount: input_v=%d tab_v=%d output_v=%d width_mm=%.3f",
        len(helmet_mesh.vertices),
        len(tab.vertices),
        len(strapped.vertices),
        strap_width_mm,
    )
    return strapped
