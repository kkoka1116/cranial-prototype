"""Tests for mesh cleanup.

We deliberately damage a synthetic head (hole, duplicate vertices, flipped
winding) and confirm clean_scan recovers a manifold + watertight mesh and
reports what it did.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from anatomy.cleanup import clean_scan
from anatomy.errors import AnatomyError
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head


def test_clean_already_good_mesh_is_noop_ish() -> None:
    head = generate_test_head(HeadConfig())
    result = clean_scan(head)
    assert result.mesh.is_watertight
    assert result.mesh.is_winding_consistent
    assert result.report.was_already_manifold is True
    assert result.report.holes_filled == 0


def test_clean_fills_hole() -> None:
    head = generate_test_head(HeadConfig())
    damaged = head.copy()
    # Remove a patch of faces to open a hole.
    keep = np.ones(len(damaged.faces), dtype=bool)
    keep[:40] = False
    damaged.update_faces(keep)
    damaged.remove_unreferenced_vertices()
    assert not damaged.is_watertight

    result = clean_scan(damaged)
    assert result.mesh.is_watertight
    assert result.mesh.is_winding_consistent
    assert result.report.holes_filled >= 1
    assert result.report.was_already_manifold is False


def test_clean_merges_duplicate_vertices() -> None:
    head = generate_test_head(HeadConfig())
    # Append a duplicate of an existing vertex (unreferenced) + keep faces.
    dup = np.vstack([head.vertices, head.vertices[0]])
    damaged = trimesh.Trimesh(vertices=dup, faces=head.faces, process=False)
    result = clean_scan(damaged)
    assert result.mesh.is_watertight
    assert result.report.duplicate_vertices_merged >= 1


def test_clean_fixes_flipped_winding() -> None:
    head = generate_test_head(HeadConfig())
    flipped = head.copy()
    flipped.faces = np.fliplr(flipped.faces)  # invert winding
    result = clean_scan(flipped)
    assert result.mesh.is_winding_consistent
    assert result.mesh.is_watertight


def test_clean_unrepairable_raises() -> None:
    # A single triangle cannot be made watertight.
    bad = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(AnatomyError):
        clean_scan(bad)
