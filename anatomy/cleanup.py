"""Mesh cleanup for raw scans.

pymeshfix does the heavy lifting (hole filling + non-manifold repair);
trimesh handles duplicate-vertex merging and degenerate-face removal. Every
change is recorded in a CleanupReport. After cleanup the mesh must pass the
*kernel's* manifold + watertight validators — that cross-package import is
the sanctioned boundary use (CLAUDE.md "Package boundaries").
"""

from __future__ import annotations

import logging

import numpy as np
import pymeshfix
import trimesh
from pydantic import BaseModel, ConfigDict

from anatomy.errors import AnatomyError
from kernel.validation.topology import check_manifold, check_watertight

logger = logging.getLogger("anatomy")


class CleanupReport(BaseModel):
    """What clean_scan changed. All counts are pre-vs-post deltas."""

    model_config = ConfigDict(frozen=True)

    was_already_manifold: bool
    duplicate_vertices_merged: int
    degenerate_faces_removed: int
    holes_filled: int
    faces_removed: int
    faces_added: int
    normals_flipped: bool
    notes: str = ""


class CleanedScan(BaseModel):
    """The cleaned mesh plus the report describing how it was cleaned."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mesh: trimesh.Trimesh
    report: CleanupReport


def _count_boundary_loops(mesh: trimesh.Trimesh) -> int:
    """Number of open boundary loops (holes). 0 for a watertight mesh."""
    if mesh.is_watertight:
        return 0
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    # Each unique boundary edge appears once; estimate loop count via the
    # edge graph's connected components over boundary edges.
    boundary_edges = mesh.edges_sorted[groups]
    if len(boundary_edges) == 0:
        return 0
    g = trimesh.graph.nx.Graph()
    g.add_edges_from(boundary_edges)
    return int(trimesh.graph.nx.number_connected_components(g))


def clean_scan(mesh: trimesh.Trimesh) -> CleanedScan:
    """Repair a raw scan into a manifold, watertight mesh.

    Pipeline: merge duplicate vertices -> drop degenerate/duplicate faces ->
    pymeshfix hole-fill & non-manifold repair -> fix winding. Raises
    AnatomyError if the result still fails the kernel's manifold/watertight
    checks (caller decides whether to abandon the scan).
    """
    # Measure the baseline from the input exactly as given — do NOT call
    # trimesh.process() first, since that silently merges/strips and would
    # zero out the deltas we need to report.
    work = mesh.copy()
    v0, f0 = len(work.vertices), len(work.faces)
    already_manifold = bool(work.is_watertight and work.is_winding_consistent)

    # 1. Merge duplicate vertices (collapses coincident referenced vertices
    #    and drops unreferenced ones).
    work.merge_vertices()
    work.remove_unreferenced_vertices()
    dup_merged = max(0, v0 - len(work.vertices))

    # 2. Remove degenerate + duplicate faces.
    f_before_degen = len(work.faces)
    work.update_faces(work.nondegenerate_faces())
    work.update_faces(work.unique_faces())
    work.remove_unreferenced_vertices()
    degenerate_removed = max(0, f_before_degen - len(work.faces))

    # 3. pymeshfix: hole fill + non-manifold repair (deterministic, no RNG).
    f_before_fix = len(work.faces)
    meshfix = pymeshfix.MeshFix(
        np.asarray(work.vertices, dtype=np.float64),
        np.asarray(work.faces, dtype=np.int64),
    )
    meshfix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
    repaired = trimesh.Trimesh(
        vertices=np.asarray(meshfix.v, dtype=np.float64),
        faces=np.asarray(meshfix.f, dtype=np.int64),
        process=False,
    )
    faces_removed = max(0, f_before_fix - len(repaired.faces))
    faces_added = max(0, len(repaired.faces) - f_before_fix)
    holes_filled = _count_boundary_loops(work)  # loops present before the fix

    # 4. Fix winding / normals.
    normals_before = repaired.is_winding_consistent
    repaired.fix_normals()
    normals_flipped = bool(not normals_before and repaired.is_winding_consistent)

    # 5. Gate on the kernel's notion of valid geometry.
    man = check_manifold(repaired)
    wat = check_watertight(repaired)
    if not (man.passed and wat.passed):
        raise AnatomyError(
            "cleanup could not produce a manifold + watertight mesh "
            f"(manifold={man.passed}, watertight={wat.passed}); "
            f"{man.message}; {wat.message}"
        )

    report = CleanupReport(
        was_already_manifold=already_manifold,
        duplicate_vertices_merged=dup_merged,
        degenerate_faces_removed=degenerate_removed,
        holes_filled=holes_filled,
        faces_removed=faces_removed,
        faces_added=faces_added,
        normals_flipped=normals_flipped,
        notes=(f"v {v0}->{len(repaired.vertices)}, f {f0}->{len(repaired.faces)}"),
    )
    logger.info(
        "scan cleaned: %s",
        report.notes,
        extra={"anatomy_event": {"operation": "clean_scan", **report.model_dump()}},
    )
    return CleanedScan(mesh=repaired, report=report)
