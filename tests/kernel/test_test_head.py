"""Tests for the synthetic test-head primitive."""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head


def test_baseline_is_manifold_and_watertight() -> None:
    mesh = generate_test_head(HeadConfig())
    assert mesh.is_winding_consistent
    assert mesh.is_volume
    assert mesh.is_watertight


def test_baseline_dimensions_match_defaults() -> None:
    mesh = generate_test_head(HeadConfig())
    extents = mesh.extents
    # Tolerances are generous because the icosphere isn't axis-aligned and
    # vertex normals slightly shift the bounding box.
    assert extents[0] == pytest.approx(130.0, abs=1.0)  # width (X)
    assert extents[1] == pytest.approx(165.0, abs=1.0)  # length (Y)
    assert extents[2] == pytest.approx(115.0, abs=1.0)  # height (Z)


def test_brachycephaly_compresses_y() -> None:
    base = generate_test_head(HeadConfig())
    brachy = generate_test_head(HeadConfig(brachycephaly_factor=0.85))
    # Y extent should be ~15% smaller.
    base_y = base.extents[1]
    brachy_y = brachy.extents[1]
    assert brachy_y == pytest.approx(0.85 * base_y, rel=0.01)
    # X and Z should be unchanged.
    assert brachy.extents[0] == pytest.approx(base.extents[0], rel=0.001)
    assert brachy.extents[2] == pytest.approx(base.extents[2], rel=0.001)


def test_occipital_flattening_is_asymmetric_when_offset() -> None:
    """A right-offset occipital indent should pull the posterior-right
    surface inward more than the posterior-left."""
    asym = generate_test_head(
        HeadConfig(
            occipital_flat_mm=10.0,
            occipital_flat_radius_mm=40.0,
            occipital_flat_lateral_offset_mm=30.0,
        )
    )
    verts = np.array(asym.vertices)
    # Sample vertices on the back (-Y), near the equator (Z near 0).
    back_mask = (verts[:, 1] < -30.0) & (np.abs(verts[:, 2]) < 20.0)
    back = verts[back_mask]
    right_back = back[back[:, 0] > 10.0]
    left_back = back[back[:, 0] < -10.0]
    assert len(right_back) > 0 and len(left_back) > 0
    # Right side should be CLOSER to origin in Y (indented).
    assert right_back[:, 1].max() > left_back[:, 1].max()


def test_generation_is_deterministic() -> None:
    cfg = HeadConfig(
        occipital_flat_mm=5.0,
        occipital_flat_lateral_offset_mm=10.0,
        brachycephaly_factor=0.9,
    )
    a = generate_test_head(cfg)
    b = generate_test_head(cfg)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)
