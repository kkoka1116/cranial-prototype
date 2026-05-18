"""Tests for the helmet shell construction."""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config import CorrectionRegion, HeadConfig, ShellConfig
from kernel.operations.shell import (
    _canonicalize_mesh,
    _correction_displacement,
    build_shell,
)
from kernel.primitives.test_head import generate_test_head


@pytest.fixture(scope="module")
def baseline_head():
    return generate_test_head(HeadConfig())


def test_shell_is_manifold_watertight_and_hollow(baseline_head) -> None:
    shell, inner, outer = build_shell(baseline_head, ShellConfig())
    assert shell.is_winding_consistent
    assert shell.is_volume
    assert shell.is_watertight
    # Hollow body: shell volume must be MUCH smaller than the outer solid.
    assert shell.volume < outer.volume / 3.0


def test_shell_wall_thickness_matches_config(baseline_head) -> None:
    """A symmetric baseline shell should have wall thickness ≈ wall_thickness_mm.

    Wall thickness = outer offset distance - inner offset distance =
    (relief + wall) - relief = wall.
    """
    cfg = ShellConfig(relief_mm=3.0, wall_thickness_mm=4.0)
    _shell, inner, outer = build_shell(baseline_head, cfg)

    # Sample inner; closest-point to outer should be ~wall_thickness.
    import trimesh

    samples, _ = trimesh.sample.sample_surface_even(inner, 1000, seed=0)
    _, distances, _ = trimesh.proximity.closest_point(outer, samples)
    assert distances.min() == pytest.approx(4.0, abs=0.1)
    assert distances.max() == pytest.approx(4.0, abs=0.1)


def test_correction_thins_wall_locally(baseline_head) -> None:
    """A correction region should locally subtract from wall thickness."""
    # The synthetic head spans z = 0 (skull base) to z = 115 (vertex) in
    # the canonical frame. Place the correction at upper-anterior cranium.
    regions = [
        CorrectionRegion(
            x_mm=0.0,
            y_mm=70.0,
            z_mm=87.5,
            radius_mm=30.0,
            magnitude_mm=1.5,
            falloff="gaussian",
        )
    ]
    cfg = ShellConfig(relief_mm=3.0, wall_thickness_mm=4.0, correction_regions=regions)
    _shell, inner, outer = build_shell(baseline_head, cfg)

    # At the correction center, wall thickness should be ~4 - 1.5 = 2.5 mm.
    # Find the inner vertex closest to the correction center, then check
    # distance to outer.
    inner_v = np.array(inner.vertices)
    target = np.array([0.0, 70.0, 87.5])
    closest_idx = int(np.argmin(np.linalg.norm(inner_v - target, axis=1)))
    nearest_point = inner_v[closest_idx]
    import trimesh

    _, dist, _ = trimesh.proximity.closest_point(outer, nearest_point.reshape(1, 3))
    # Center magnitude is full 1.5 mm, so wall ≈ 2.5 mm.
    assert dist[0] == pytest.approx(2.5, abs=0.2)


def test_falloff_modes_all_work() -> None:
    # Sample at center (d=0), at the radius (d=50), and far out (d=200 = 4 sigma).
    verts = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    for mode in ("gaussian", "cosine", "linear"):
        region = CorrectionRegion(
            x_mm=0.0,
            y_mm=0.0,
            z_mm=0.0,
            radius_mm=50.0,
            magnitude_mm=2.0,
            falloff=mode,
        )
        disp = _correction_displacement(verts, region)
        # All falloffs are full magnitude at the center.
        assert disp[0] == pytest.approx(2.0, abs=0.01)
        if mode == "gaussian":
            # At 1 sigma (d=50), Gaussian = e^(-0.5) ≈ 0.607 → 2.0 * 0.607 ≈ 1.21.
            assert disp[1] == pytest.approx(2.0 * np.exp(-0.5), abs=0.01)
            # At 4 sigma, weight is tiny.
            assert disp[2] < 0.01
        elif mode == "cosine":
            # At the radius, raised-cosine weight = 0.5(1+cos(π)) = 0.
            assert disp[1] == pytest.approx(0.0, abs=1e-9)
            assert disp[2] == pytest.approx(0.0, abs=1e-9)
        elif mode == "linear":
            # At the radius, linear weight = 0.
            assert disp[1] == pytest.approx(0.0, abs=1e-9)
            assert disp[2] == pytest.approx(0.0, abs=1e-9)


def test_build_shell_raises_on_bad_input() -> None:
    """A degenerate head input (single triangle) is not a valid volume — must raise."""
    import trimesh

    from kernel.errors import GeometryError

    bad = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(GeometryError, match="not a valid volume"):
        build_shell(bad, ShellConfig())


def test_canonicalize_mesh_is_idempotent() -> None:
    """Canonicalization applied twice == applied once."""
    verts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]], dtype=np.int64)
    v1, f1 = _canonicalize_mesh(verts, faces)
    v2, f2 = _canonicalize_mesh(v1, f1)
    assert np.array_equal(v1, v2)
    assert np.array_equal(f1, f2)


def test_canonicalize_normalizes_shuffled_input() -> None:
    """Two meshes with the same geometry but different vertex order
    canonicalize to the same representation."""
    rng = np.random.default_rng(42)
    verts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]], dtype=np.int64)

    perm = rng.permutation(4)
    inv = np.empty(4, dtype=np.int64)
    inv[perm] = np.arange(4)
    verts_shuffled = verts[perm]
    faces_shuffled = inv[faces]

    v1, f1 = _canonicalize_mesh(verts, faces)
    v2, f2 = _canonicalize_mesh(verts_shuffled, faces_shuffled)
    assert np.array_equal(v1, v2)
    assert np.array_equal(f1, f2)
