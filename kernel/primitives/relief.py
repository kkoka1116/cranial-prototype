"""Local relief-zone carving on a finished helmet shell.

A *relief zone* gives the patient's head **more** clearance over a localized
region — for example over a frontal bossing, so the helmet does not press on
the prominence. We implement it by locally deepening the helmet's head-facing
inner cavity: a smooth radial "tool" solid is positioned against the inner
(cavity) wall and boolean-subtracted from the shell. The tool removes up to
``depth_mm`` of wall material at ``center_mm``, tapering smoothly to zero by
``radius_mm`` according to a ``falloff`` profile ("gaussian" | "linear" |
"cosine").

This is an *additive* Week-4-permitted kernel primitive: it takes a
finished, watertight helmet shell (e.g. from
:func:`kernel.operations.shell.build_shell`) and returns a new watertight
shell. It never mutates its input and contains zero AI / no decisions —
caller-supplied parameters in, deterministic geometry out.

Tool-solid design
-----------------
The tool is a closed "profiled dome": a triangulated cap whose top surface
height above its flat base follows ``depth_mm * weight(d)`` where ``d`` is the
in-plane distance from the projected ``center_mm`` and ``weight`` is the
chosen falloff (1 at the center, 0 at ``radius_mm``). The dome is oriented
along the local *outward* axis (from the shell centroid toward ``center_mm``)
and recessed behind the cavity wall by a fixed base depth so that subtracting
it carves the inner wall *outward* (away from the head) by ``depth_mm`` at the
center. The topology uses an explicit apex vertex (no polar-grid singularity),
so the tool is itself manifold + watertight and the boolean is robust.

Sign of the volume change
-------------------------
``trimesh.Trimesh.volume`` for the hollow shell (an ``outer - inner``
manifold) is the volume of the **solid wall material**. A relief removes wall
material from the cavity side, so the relieved shell's ``volume`` is *strictly
smaller* than the input shell's ``volume`` (the enclosed empty cavity grows;
the solid shrinks). A larger ``depth_mm`` removes more material, so the volume
decreases monotonically with ``depth_mm``.

Determinism
-----------
After the manifold3d boolean we convert back via
:func:`kernel.operations.shell._from_manifold`, which canonicalizes the
vertex / face ordering (lexsort + face rotation/sort). We never call
``trimesh.fix_normals`` on the boolean output (per the repo determinism note);
``fix_normals`` *is* used on the tool we construct ourselves, which is
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

# Tool tessellation. Fixed so the tool topology — and therefore the boolean
# output — is identical across runs (determinism, invariant #2).
_TOOL_RINGS = 24
_TOOL_SECTORS = 64

# How far (mm) the tool's flat base sits *behind* ``center_mm`` along the
# inward axis, i.e. recessed into the cavity. Must comfortably exceed any
# realistic relief depth so the tool fully spans the inner-wall region and the
# subtraction is a clean carve rather than a thin sliver.
_TOOL_BASE_RECESS_MM = 12.0

# Safety margin (mm) kept between the relief depth and the local wall
# thickness. A relief deeper than (wall_thickness - margin) would perforate
# the shell; we reject it before the boolean rather than emit broken geometry.
_PERFORATION_MARGIN_MM = 0.5

_VALID_FALLOFFS = ("gaussian", "linear", "cosine")


def _falloff_weight(distances_mm: np.ndarray, radius_mm: float, falloff: str) -> np.ndarray:
    """Radial weight in ``[0, 1]``: 1 at distance 0, 0 at/after ``radius_mm``.

    Mirrors the falloff conventions used elsewhere in the kernel
    (``kernel.operations.shell._correction_displacement``).
    """
    if falloff == "gaussian":
        # 1-sigma = radius/3 so the profile is ~0 by one radius out. We offset
        # and renormalize so the weight is exactly 1 at d=0 and exactly 0 at
        # d=radius (a bare Gaussian never reaches 0 — that would leave a
        # discontinuous lip at the tool rim and risk a non-manifold boolean).
        sigma = radius_mm / 3.0
        g = np.exp(-(distances_mm**2) / (2.0 * sigma**2))
        g_rim = float(np.exp(-(radius_mm**2) / (2.0 * sigma**2)))
        weight = (g - g_rim) / (1.0 - g_rim)
    elif falloff == "linear":
        weight = 1.0 - distances_mm / radius_mm
    elif falloff == "cosine":
        normalized = np.clip(distances_mm / radius_mm, 0.0, 1.0)
        weight = 0.5 * (1.0 + np.cos(np.pi * normalized))
    else:  # pragma: no cover — guarded by _validate_args before we get here
        raise GeometryError(f"unknown falloff: {falloff}")
    return np.clip(weight, 0.0, 1.0)


def _local_outward_axis(
    shell_mesh: trimesh.Trimesh, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an orthonormal frame ``(u, v, axis)`` for the relief tool.

    ``axis`` is the unit vector from the shell centroid to ``center`` (the
    local "outward", head-away direction). ``u`` and ``v`` span the plane
    perpendicular to ``axis``. The frame is derived deterministically from the
    geometry, with a fixed tie-breaking reference vector.
    """
    axis = center - np.asarray(shell_mesh.centroid, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise GeometryError(
            "relief center coincides with the shell centroid; " "cannot derive an outward axis",
            context={"center_mm": center.tolist()},
        )
    axis = axis / norm
    # Fixed tie-breaking reference so the in-plane frame is reproducible.
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v, axis


def _local_wall_thickness_mm(
    shell_mesh: trimesh.Trimesh, center: np.ndarray, axis: np.ndarray
) -> float | None:
    """Measure the shell's solid-wall thickness at ``center`` along ``axis``.

    Casts a ray from well inside the cavity, through ``center``, in the
    outward direction. The first two hits are the inner then the outer wall;
    their separation is the local wall thickness. Returns ``None`` if the ray
    does not cleanly cross the wall (caller treats that as "cannot prove it is
    safe" and rejects).
    """
    origin = center - axis * (2.0 * _TOOL_BASE_RECESS_MM + 50.0)
    locations, _, _ = shell_mesh.ray.intersects_location(
        ray_origins=np.array([origin]),
        ray_directions=np.array([axis]),
    )
    if len(locations) < 2:
        return None
    params = sorted(float(np.dot(loc - origin, axis)) for loc in locations)
    return params[1] - params[0]


def _build_relief_tool(
    center: np.ndarray,
    radius_mm: float,
    depth_mm: float,
    u: np.ndarray,
    v: np.ndarray,
    axis: np.ndarray,
    falloff: str,
) -> trimesh.Trimesh:
    """Construct the closed, watertight profiled-dome subtraction tool.

    Topology (deterministic; ``R = _TOOL_RINGS``, ``S = _TOOL_SECTORS``):

        - ``R`` concentric rings of ``S`` vertices on the *top* (domed)
          surface, radius stepping from ``radius/R`` out to ``radius``.
        - ``R`` matching rings on the flat *base* plane (recessed behind
          ``center`` by ``_TOOL_BASE_RECESS_MM`` along ``-axis``).
        - one apex vertex on top, one on the base, closing the center with a
          triangle fan (no polar-coordinate degeneracy).
        - an outer cylindrical rim band joining the last top and base rings.
    """
    sectors = _TOOL_SECTORS
    rings = _TOOL_RINGS
    radii = np.linspace(0.0, radius_mm, rings + 1)[1:]
    angles = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    top_verts: list[np.ndarray] = []
    base_verts: list[np.ndarray] = []
    for r in radii:
        weight = _falloff_weight(np.array([r]), radius_mm, falloff)[0]
        height = depth_mm * weight
        for ca, sa in zip(cos_a, sin_a, strict=True):
            planar = center + (r * ca) * u + (r * sa) * v
            top_verts.append(planar + axis * height)
            base_verts.append(planar - axis * _TOOL_BASE_RECESS_MM)

    n_ring = rings * sectors
    verts = np.array(
        [
            *top_verts,
            *base_verts,
            center + axis * depth_mm,  # top apex
            center - axis * _TOOL_BASE_RECESS_MM,  # base apex
        ],
        dtype=np.float64,
    )
    apex_top = 2 * n_ring
    apex_base = 2 * n_ring + 1

    def top(ring: int, sec: int) -> int:
        return ring * sectors + (sec % sectors)

    def base(ring: int, sec: int) -> int:
        return n_ring + ring * sectors + (sec % sectors)

    faces: list[list[int]] = []

    # Center fans (innermost ring 0). Top normal points along +axis (outward);
    # base normal points along -axis (the opposite winding).
    for s in range(sectors):
        faces.append([apex_top, top(0, s), top(0, s + 1)])
        faces.append([apex_base, base(0, s + 1), base(0, s)])

    # Top and base surface bands between successive rings.
    for ring in range(rings - 1):
        for s in range(sectors):
            faces.append([top(ring, s), top(ring + 1, s), top(ring + 1, s + 1)])
            faces.append([top(ring, s), top(ring + 1, s + 1), top(ring, s + 1)])
            faces.append([base(ring, s), base(ring + 1, s + 1), base(ring + 1, s)])
            faces.append([base(ring, s), base(ring, s + 1), base(ring + 1, s + 1)])

    # Outer rim wall joining the last top ring to the last base ring.
    last = rings - 1
    for s in range(sectors):
        faces.append([top(last, s), base(last, s), base(last, s + 1)])
        faces.append([top(last, s), base(last, s + 1), top(last, s + 1)])

    tool = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=False)
    # fix_normals is empirically deterministic for the pinned trimesh version
    # on a mesh we constructed ourselves (same rationale as
    # kernel/operations/trim.py). It is NOT applied to the boolean output.
    tool.fix_normals()
    return tool


def _validate_args(
    helmet_mesh: trimesh.Trimesh,
    center_mm: tuple[float, float, float],
    radius_mm: float,
    depth_mm: float,
    falloff: str,
) -> None:
    """Reject malformed inputs before touching geometry."""
    if not helmet_mesh.is_volume:
        raise GeometryError(
            "helmet input to apply_relief is not a valid volume",
            context={
                "is_winding_consistent": bool(helmet_mesh.is_winding_consistent),
                "is_watertight": bool(helmet_mesh.is_watertight),
            },
        )
    if radius_mm <= 0.0:
        raise GeometryError(f"relief radius_mm must be > 0, got {radius_mm}")
    if depth_mm <= 0.0:
        raise GeometryError(f"relief depth_mm must be > 0, got {depth_mm}")
    if falloff not in _VALID_FALLOFFS:
        raise GeometryError(f"unknown falloff {falloff!r}; expected one of {_VALID_FALLOFFS}")


def apply_relief(
    helmet_mesh: trimesh.Trimesh,
    center_mm: tuple[float, float, float],
    radius_mm: float,
    depth_mm: float,
    falloff: str = "gaussian",
) -> trimesh.Trimesh:
    """Carve a smooth local relief zone into a finished helmet shell.

    Args:
        helmet_mesh: A watertight, winding-consistent helmet shell in the
            canonical reference frame (origin between the ear canals, +X
            patient right, +Y anterior, +Z superior). Not mutated.
        center_mm: ``(x, y, z)`` of the relief center, in mm, in the canonical
            frame. Should lie on / near the inner cavity wall.
        radius_mm: Influence radius in mm. The relief depth tapers from
            ``depth_mm`` at the center to 0 at this distance.
        depth_mm: Maximum extra clearance carved at the center, in mm. Must be
            comfortably less than the local wall thickness or the shell would
            perforate (raises ``GeometryError``).
        falloff: Radial profile — ``"gaussian"`` (default), ``"linear"``, or
            ``"cosine"``.

    Returns:
        A new ``trimesh.Trimesh`` — the relieved shell, watertight and
        winding-consistent. Its ``volume`` (solid wall material) is strictly
        smaller than the input's because material was removed from the cavity.

    Raises:
        kernel.errors.GeometryError: if the input is not a valid volume, the
            arguments are out of range, the relief would perforate the wall
            (``depth_mm`` ≥ local wall thickness − margin), or the boolean
            produced empty / invalid geometry.
    """
    _validate_args(helmet_mesh, center_mm, radius_mm, depth_mm, falloff)

    center = np.array(center_mm, dtype=np.float64)
    u, v, axis = _local_outward_axis(helmet_mesh, center)

    # Perforation guard (invariant #6: never emit broken geometry). A relief
    # as deep as the local wall would punch a hole clean through the shell.
    wall_mm = _local_wall_thickness_mm(helmet_mesh, center, axis)
    if wall_mm is None:
        raise GeometryError(
            "could not measure local wall thickness at the relief center; "
            "the center may be off the cavity wall",
            context={"center_mm": center.tolist()},
        )
    if depth_mm >= wall_mm - _PERFORATION_MARGIN_MM:
        raise GeometryError(
            f"relief depth_mm={depth_mm} would perforate the shell at "
            f"center {center.tolist()} (local wall thickness "
            f"{wall_mm:.3f} mm, margin {_PERFORATION_MARGIN_MM} mm)",
            context={
                "center_mm": center.tolist(),
                "depth_mm": float(depth_mm),
                "local_wall_thickness_mm": float(wall_mm),
            },
        )

    tool = _build_relief_tool(center, radius_mm, depth_mm, u, v, axis, falloff)
    if not tool.is_volume:
        raise GeometryError(
            "constructed relief tool is not a valid volume",
            context={
                "is_watertight": bool(tool.is_watertight),
                "is_winding_consistent": bool(tool.is_winding_consistent),
                "n_vertices": int(len(tool.vertices)),
            },
        )

    helmet_m = _to_manifold(helmet_mesh)
    tool_m = _to_manifold(tool)
    relieved_m = helmet_m - tool_m
    if relieved_m.is_empty():
        raise GeometryError(
            "relief boolean subtract produced an empty manifold",
            context={"center_mm": center.tolist(), "depth_mm": float(depth_mm)},
        )

    relieved = _from_manifold(relieved_m)

    # Final invariant gate: never let broken geometry escape the kernel.
    if not relieved.is_winding_consistent or not relieved.is_watertight:
        raise GeometryError(
            "relief boolean produced non-watertight / non-winding-consistent "
            "geometry (likely a perforation slipped past the depth guard)",
            context={
                "center_mm": center.tolist(),
                "depth_mm": float(depth_mm),
                "is_watertight": bool(relieved.is_watertight),
                "is_winding_consistent": bool(relieved.is_winding_consistent),
            },
        )

    logger.debug(
        "apply_relief: input_v=%d tool_v=%d output_v=%d depth_mm=%.3f",
        len(helmet_mesh.vertices),
        len(tool.vertices),
        len(relieved.vertices),
        depth_mm,
    )
    return relieved
