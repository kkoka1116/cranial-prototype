"""Helmet shell construction.

The shell is built from two offset surfaces of the head:

    outer surface = head offset outward by (relief_mm + wall_thickness_mm),
                    minus per-region inward offsets (correction zones).
    inner surface = head offset outward by relief_mm.

The helmet shell = outer − inner (boolean subtraction).

Offsets are computed as per-vertex displacements along the vertex normals
of the head mesh. This is a *naïve* offset (not a true Minkowski sum) and
will introduce small self-intersections at high-curvature regions if the
offset distance is large compared to the local curvature radius. For the
head-sized geometry and offset distances used in week 1 (relief ~3 mm,
wall ~4 mm, on a 60–80 mm radius surface), the resulting meshes have been
empirically stable. manifold3d's boolean subtract is robust against
small numerical issues in the input.

For correction regions: each adds an inward displacement to the outer
surface only (the inner surface stays a simple offset of the head + relief
gap). The result is that the helmet wall *thins* where corrections are
applied, which is the intended biomechanical effect — the helmet contacts
the head with light pressure on the prominences we want to remodel.
"""

from __future__ import annotations

import logging

import manifold3d
import numpy as np
import trimesh

from kernel.config import CorrectionRegion, ShellConfig
from kernel.errors import GeometryError

logger = logging.getLogger(__name__)


def _correction_displacement(
    vertices: np.ndarray,
    region: CorrectionRegion,
) -> np.ndarray:
    """Per-vertex scalar inward displacement contribution from one region."""
    center = np.array([region.x_mm, region.y_mm, region.z_mm], dtype=np.float64)
    d = np.linalg.norm(vertices - center, axis=1)
    if region.falloff == "gaussian":
        weight = np.exp(-(d**2) / (2.0 * region.radius_mm**2))
    elif region.falloff == "cosine":
        # Smooth raised cosine inside the radius, zero outside.
        normalized = np.clip(d / region.radius_mm, 0.0, 1.0)
        weight = 0.5 * (1.0 + np.cos(np.pi * normalized))
        weight = np.where(d <= region.radius_mm, weight, 0.0)
    elif region.falloff == "linear":
        weight = np.clip(1.0 - d / region.radius_mm, 0.0, 1.0)
    else:  # pragma: no cover — pydantic prevents this
        raise GeometryError(f"unknown falloff: {region.falloff}")
    return weight * region.magnitude_mm


def _offset_mesh(
    head_mesh: trimesh.Trimesh,
    base_offset_mm: float,
    correction_regions: list[CorrectionRegion] | None = None,
) -> trimesh.Trimesh:
    """Offset the head surface outward by base_offset_mm, with optional
    per-vertex inward corrections subtracted from the offset.

    Preserves face topology (every vertex displaces along its own normal),
    so the offset mesh has the same winding as the input. The input head
    mesh is assumed to have correct outward normals (the synthetic head
    primitive ensures this via its own fix_normals pass), so we don't
    re-fix here — re-fixing would be a no-op on a well-wound mesh.
    """
    verts = np.array(head_mesh.vertices, dtype=np.float64)
    normals = np.array(head_mesh.vertex_normals, dtype=np.float64)

    # Net outward displacement = base_offset - sum_of_corrections (each
    # correction is an inward displacement on the outer surface).
    net = np.full(verts.shape[0], base_offset_mm, dtype=np.float64)
    if correction_regions:
        for region in correction_regions:
            net -= _correction_displacement(verts, region)

    new_verts = verts + normals * net[:, None]
    return trimesh.Trimesh(vertices=new_verts, faces=head_mesh.faces, process=False)


def _to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """Convert a trimesh.Trimesh to a manifold3d.Manifold."""
    mesh_obj = manifold3d.Mesh(
        vert_properties=np.array(mesh.vertices, dtype=np.float32),
        tri_verts=np.array(mesh.faces, dtype=np.uint32),
    )
    return manifold3d.Manifold(mesh_obj)


def _canonicalize_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder vertices and faces into a canonical form.

    manifold3d's boolean output is geometrically deterministic but the
    vertex / face ordering is NOT — it depends on internal parallel
    scheduling. Two runs on the same inputs produce the same vertex set
    but in different orders. We canonicalize to make the byte output
    reproducible (invariant #2):

      1. Sort vertices lexicographically by (x, y, z).
      2. Remap face indices to the new vertex order.
      3. Rotate each face so the smallest index is first (this preserves
         winding because rotation is a cyclic, even permutation on 3
         elements).
      4. Sort faces lexicographically by their vertex tuples.

    Returns (canonical_verts, canonical_faces).
    """
    # 1. Lex sort of vertices by (x, y, z) — use lexsort with z as the
    # primary, y secondary, x tertiary so the sort key matches np.lexsort's
    # last-key-first convention.
    order = np.lexsort((verts[:, 2], verts[:, 1], verts[:, 0]))
    new_verts = verts[order]

    # inv[i] = new index of old vertex i
    inv = np.empty(verts.shape[0], dtype=np.int64)
    inv[order] = np.arange(verts.shape[0], dtype=np.int64)

    # 2. Remap faces.
    new_faces = inv[faces]

    # 3. Rotate each face so the minimum index is first. Use a cyclic
    # rotation (preserves winding).
    min_idx = np.argmin(new_faces, axis=1)
    rotated = np.empty_like(new_faces)
    for r in range(3):
        mask = min_idx == r
        rotated[mask] = np.roll(new_faces[mask], -r, axis=1)
    new_faces = rotated

    # 4. Sort faces lexicographically.
    face_order = np.lexsort((new_faces[:, 2], new_faces[:, 1], new_faces[:, 0]))
    new_faces = new_faces[face_order]

    return new_verts, new_faces


def _from_manifold(m: manifold3d.Manifold) -> trimesh.Trimesh:
    """Convert a manifold3d.Manifold back to a trimesh.Trimesh.

    Notes:
        - We do NOT call trimesh.fix_normals on boolean output. manifold3d
          already produces consistent outward-facing winding, and on a
          hollow body fix_normals can mis-orient interior faces (turning
          outer−inner into outer+inner by flipping winding).
        - We DO canonicalize vertex/face ordering: manifold3d's boolean
          output is geometrically deterministic but its vertex order is
          parallel-scheduling-dependent, so we sort to make the byte output
          reproducible (invariant #2).
    """
    mesh_obj = m.to_mesh()
    verts = np.array(mesh_obj.vert_properties, dtype=np.float64)
    # vert_properties may include extra channels; the first 3 are XYZ.
    verts = verts[:, :3]
    faces = np.array(mesh_obj.tri_verts, dtype=np.int64)
    verts, faces = _canonicalize_mesh(verts, faces)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_shell(
    head_mesh: trimesh.Trimesh,
    config: ShellConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """Construct the helmet shell from a head mesh.

    Returns:
        (helmet_shell, inner_surface_mesh, outer_surface_mesh)
            helmet_shell        — the full hollow helmet, manifold + watertight.
            inner_surface_mesh  — the inner offset of the head (head + relief).
            outer_surface_mesh  — the outer offset of the head (head + relief +
                                  wall_thickness − corrections).
        Returning both surface meshes lets the wall-thickness validator
        measure distance directly from inner to outer without picking
        ambiguous closest-points on the merged helmet mesh.
    """
    outer = _offset_mesh(
        head_mesh,
        base_offset_mm=config.relief_mm + config.wall_thickness_mm,
        correction_regions=list(config.correction_regions),
    )
    inner = _offset_mesh(
        head_mesh,
        base_offset_mm=config.relief_mm,
        correction_regions=None,
    )

    # Sanity: both must be manifold + watertight inputs for the boolean.
    if not outer.is_volume:
        raise GeometryError(
            "outer offset mesh is not a valid volume",
            context={
                "is_winding_consistent": bool(outer.is_winding_consistent),
                "is_watertight": bool(outer.is_watertight),
            },
        )
    if not inner.is_volume:
        raise GeometryError(
            "inner offset mesh is not a valid volume",
            context={
                "is_winding_consistent": bool(inner.is_winding_consistent),
                "is_watertight": bool(inner.is_watertight),
            },
        )

    # Boolean: shell = outer − inner.
    outer_m = _to_manifold(outer)
    inner_m = _to_manifold(inner)
    shell_m = outer_m - inner_m

    if shell_m.is_empty():
        raise GeometryError(
            "shell boolean subtract produced an empty manifold",
            context={
                "outer_volume_mm3": float(outer.volume),
                "inner_volume_mm3": float(inner.volume),
            },
        )

    shell = _from_manifold(shell_m)
    logger.debug(
        "build_shell: outer_v=%d inner_v=%d shell_v=%d",
        len(outer.vertices),
        len(inner.vertices),
        len(shell.vertices),
    )
    return shell, inner, outer
