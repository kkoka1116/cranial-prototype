"""Trim-line application.

The trim line is defined as a parametric closed curve in (azimuth, elevation).
For each azimuth θ around the head, we interpolate the control points to a
Z height:

    z_cut(θ) = z_centroid + R_proj * sin(elevation(θ))

where R_proj is the helmet's bounding-sphere radius (so elevation values are
scale-invariant with the helmet).

Cutting body geometry — a "wavy donut" (torus-like prism with rectangular
cross-section):

    Top outer rim (TOR):  (R_outer sin θ, R_outer cos θ, z_cut(θ))
    Top inner rim (TIR):  (R_inner sin θ, R_inner cos θ, z_cut(θ))
    Bot outer rim (BOR):  (R_outer sin θ, R_outer cos θ, z_floor)
    Bot inner rim (BIR):  (R_inner sin θ, R_inner cos θ, z_floor)

    R_inner is small (1 mm) so the donut's inner hole is a tiny cylinder on
    the Z axis. R_outer is well outside the helmet so the donut fully
    encloses the helmet shell laterally.

Surfaces (each closes a face of the rectangular cross-section as we sweep
around in azimuth):

    Top annulus    : TIR → TOR  (with z varying as z_cut(θ))
    Outer wall     : TOR → BOR  (vertical, wavy top, flat bottom)
    Bottom annulus : BIR → BOR  (flat at z_floor)
    Inner wall     : TIR → BIR  (tiny inner cylinder)

This is a closed manifold. The donut's interior includes everything between
R_inner and R_outer, between z_floor and z_cut(θ). Since the helmet shell
sits at r ≈ 70–90 mm (between R_inner and R_outer), helmet material below
z_cut(θ) is inside the cutter and gets subtracted. Helmet material ABOVE
z_cut(θ) is outside the donut and is preserved.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from kernel.config import TrimConfig, TrimControlPoint
from kernel.errors import GeometryError
from kernel.operations.shell import _from_manifold, _to_manifold

logger = logging.getLogger(__name__)

# Number of azimuth samples used to discretize the trim curve. Fixed so the
# cutting body has the same topology across runs (determinism).
TRIM_SAMPLES = 240


def _sorted_control_points(
    points: list[TrimControlPoint],
) -> list[TrimControlPoint]:
    """Return control points sorted by azimuth, all wrapped into [0, 360)."""
    normalized = [
        TrimControlPoint(
            azimuth_deg=cp.azimuth_deg % 360.0,
            elevation_deg=cp.elevation_deg,
        )
        for cp in points
    ]
    return sorted(normalized, key=lambda cp: cp.azimuth_deg)


def _interpolate_elevation(sorted_points: list[TrimControlPoint], azimuth_deg: float) -> float:
    """Wrap-around linear interpolation of elevation at the given azimuth."""
    az = azimuth_deg % 360.0
    n = len(sorted_points)
    for i in range(n):
        a0 = sorted_points[i].azimuth_deg
        a1 = sorted_points[(i + 1) % n].azimuth_deg
        # Walk forward around the circle from a0 to a1 (mod 360).
        a1_eff = a1 + 360.0 if a1 < a0 else a1
        az_eff = az if az >= a0 else az + 360.0
        if a0 <= az_eff <= a1_eff:
            t = 0.0 if a1_eff == a0 else (az_eff - a0) / (a1_eff - a0)
            e0 = sorted_points[i].elevation_deg
            e1 = sorted_points[(i + 1) % n].elevation_deg
            return float(e0 + t * (e1 - e0))
    raise GeometryError(  # pragma: no cover
        f"trim interpolation failed at azimuth {azimuth_deg}"
    )


def _build_cutting_body(
    z_cuts: np.ndarray,
    azimuths_rad: np.ndarray,
    r_outer_mm: float,
    r_inner_mm: float,
    z_floor_mm: float,
) -> trimesh.Trimesh:
    """Build a manifold "wavy donut" cutting body.

    Layout (n = len(z_cuts) = len(azimuths_rad)):
        verts[0 .. n)         TIR (top inner rim, at z_cut(θ))
        verts[n .. 2n)        TOR (top outer rim, at z_cut(θ))
        verts[2n .. 3n)       BIR (bottom inner rim, at z_floor)
        verts[3n .. 4n)       BOR (bottom outer rim, at z_floor)
    """
    n = z_cuts.shape[0]
    sin_a = np.sin(azimuths_rad)
    cos_a = np.cos(azimuths_rad)

    tir = np.column_stack([r_inner_mm * sin_a, r_inner_mm * cos_a, z_cuts])
    tor = np.column_stack([r_outer_mm * sin_a, r_outer_mm * cos_a, z_cuts])
    bir = np.column_stack([r_inner_mm * sin_a, r_inner_mm * cos_a, np.full(n, z_floor_mm)])
    bor = np.column_stack([r_outer_mm * sin_a, r_outer_mm * cos_a, np.full(n, z_floor_mm)])

    verts = np.vstack([tir, tor, bir, bor])

    def TIR(k: int) -> int:  # noqa: N802 — local alias for readability
        return k % n

    def TOR(k: int) -> int:  # noqa: N802
        return n + (k % n)

    def BIR(k: int) -> int:  # noqa: N802
        return 2 * n + (k % n)

    def BOR(k: int) -> int:  # noqa: N802
        return 3 * n + (k % n)

    faces: list[list[int]] = []

    # Top annulus: quad (TIR_k, TOR_k, TOR_{k+1}, TIR_{k+1}). Outward normal
    # points upward (+Z) since this is the top face.
    for k in range(n):
        a, b = TIR(k), TIR(k + 1)
        c, d = TOR(k), TOR(k + 1)
        faces.append([a, c, d])
        faces.append([a, d, b])

    # Bottom annulus: same vertices on the floor. Outward normal -Z (down).
    # Reverse winding compared to top.
    for k in range(n):
        a, b = BIR(k), BIR(k + 1)
        c, d = BOR(k), BOR(k + 1)
        faces.append([a, d, c])
        faces.append([a, b, d])

    # Outer wall: between TOR and BOR. Outward normal points radially out.
    for k in range(n):
        a, b = TOR(k), TOR(k + 1)
        c, d = BOR(k), BOR(k + 1)
        # Winding so the outward normal points radially outward (away from
        # the Z axis). Quad order: TOR_k -> BOR_k -> BOR_{k+1} -> TOR_{k+1}.
        faces.append([a, c, d])
        faces.append([a, d, b])

    # Inner wall: between TIR and BIR. Outward normal points radially inward
    # (toward the Z axis), so winding is reversed vs. the outer wall.
    for k in range(n):
        a, b = TIR(k), TIR(k + 1)
        c, d = BIR(k), BIR(k + 1)
        faces.append([a, d, c])
        faces.append([a, b, d])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def apply_trim(
    helmet_mesh: trimesh.Trimesh,
    config: TrimConfig,
) -> trimesh.Trimesh:
    """Apply the trim cut to the helmet, returning a manifold + watertight shell.

    Cuts away helmet material where z < z_cut(θ), with z_cut interpolated
    from the control points.
    """
    if not helmet_mesh.is_volume:
        raise GeometryError(
            "helmet input to apply_trim is not a valid volume",
            context={
                "is_winding_consistent": bool(helmet_mesh.is_winding_consistent),
                "is_watertight": bool(helmet_mesh.is_watertight),
            },
        )

    sorted_pts = _sorted_control_points(list(config.control_points))

    # Project radius for elevation -> Z mapping. We use the helmet's bounding
    # sphere radius so elevation values are scale-invariant with the helmet.
    centroid = helmet_mesh.bounds.mean(axis=0)
    bbox_radius = float(0.5 * np.linalg.norm(helmet_mesh.extents))
    bbox_diag = float(np.linalg.norm(helmet_mesh.extents))

    # Sample azimuths uniformly and interpolate elevation -> z_cut.
    azimuths_deg = np.linspace(0.0, 360.0, TRIM_SAMPLES, endpoint=False)
    z_cuts = np.zeros(TRIM_SAMPLES, dtype=np.float64)
    for k, az_deg in enumerate(azimuths_deg):
        el_deg = _interpolate_elevation(sorted_pts, float(az_deg))
        z_cuts[k] = centroid[2] + bbox_radius * np.sin(np.deg2rad(el_deg))
    azimuths_rad = np.deg2rad(azimuths_deg)

    # Outer wall: well outside the helmet so the boolean is robust.
    r_outer = bbox_diag * 2.0
    # Inner hole: tiny radius — just enough to be a non-degenerate ring.
    # The helmet shell never reaches the Z axis, so 1 mm is safe.
    r_inner = 1.0
    # Floor: below the lowest z_cut by cut_depth_mm.
    z_floor = float(np.min(z_cuts)) - config.cut_depth_mm

    cutter = _build_cutting_body(
        z_cuts=z_cuts,
        azimuths_rad=azimuths_rad,
        r_outer_mm=r_outer,
        r_inner_mm=r_inner,
        z_floor_mm=z_floor,
    )

    if not cutter.is_volume:
        raise GeometryError(
            "constructed trim cutting body is not a valid volume",
            context={
                "n_vertices": int(len(cutter.vertices)),
                "n_faces": int(len(cutter.faces)),
                "is_watertight": bool(cutter.is_watertight),
                "is_winding_consistent": bool(cutter.is_winding_consistent),
            },
        )

    helmet_m = _to_manifold(helmet_mesh)
    cutter_m = _to_manifold(cutter)
    trimmed_m = helmet_m - cutter_m
    if trimmed_m.is_empty():
        raise GeometryError("trim boolean produced an empty manifold")

    trimmed = _from_manifold(trimmed_m)
    logger.debug(
        "apply_trim: input_v=%d cutter_v=%d output_v=%d",
        len(helmet_mesh.vertices),
        len(cutter.vertices),
        len(trimmed.vertices),
    )
    return trimmed
