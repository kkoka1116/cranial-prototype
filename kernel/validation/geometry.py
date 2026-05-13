"""Geometric validators: wall thickness, mass estimate, bounding box."""

from __future__ import annotations

import numpy as np
import trimesh

from kernel.validation.topology import ValidationResult


def check_min_wall_thickness(
    inner_surface_mesh: trimesh.Trimesh,
    outer_surface_mesh: trimesh.Trimesh,
    min_thickness_mm: float,
    sample_count: int = 4000,
) -> ValidationResult:
    """Estimate the minimum wall thickness of the helmet shell.

    Sample points on the inner-offset surface (the side that contacts
    head + relief) and measure the perpendicular distance to the outer-
    offset surface. The minimum across all samples is the worst-case
    wall thickness.

    Determinism: trimesh.sample.sample_surface_even uses a fixed seed.
    trimesh.proximity.closest_point's BVH query is deterministic per
    pinned trimesh version.
    """
    if min_thickness_mm <= 0:
        raise ValueError("min_thickness_mm must be positive")

    rng_seed = 0
    samples, _ = trimesh.sample.sample_surface_even(
        inner_surface_mesh, sample_count, seed=rng_seed
    )
    if samples.shape[0] == 0:
        samples, _ = trimesh.sample.sample_surface(
            inner_surface_mesh, sample_count, seed=rng_seed
        )

    _, distances, _ = trimesh.proximity.closest_point(outer_surface_mesh, samples)
    actual_min = float(np.min(distances))
    passed = actual_min >= min_thickness_mm

    return ValidationResult(
        metric_name="min_wall_thickness_mm",
        passed=passed,
        actual_value=actual_min,
        limit_value=min_thickness_mm,
        message=(
            f"min wall thickness {actual_min:.2f} mm "
            f"({'>=' if passed else '<'} limit {min_thickness_mm:.2f} mm)"
        ),
    )


def estimate_mass_g(
    mesh: trimesh.Trimesh, density_g_per_cm3: float
) -> float:
    """Estimate mass in grams from mesh volume and material density.

    trimesh.volume is in the cube of the mesh's units. Our units are mm,
    so volume_mm3 → cm3 via /1000.
    """
    if density_g_per_cm3 <= 0:
        raise ValueError("density must be positive")
    volume_mm3 = float(mesh.volume)
    volume_cm3 = volume_mm3 / 1000.0
    return volume_cm3 * density_g_per_cm3


def check_mass(
    mesh: trimesh.Trimesh,
    density_g_per_cm3: float,
    max_mass_g: float,
) -> ValidationResult:
    """Pass if estimated mass is below the limit."""
    mass = estimate_mass_g(mesh, density_g_per_cm3)
    passed = mass <= max_mass_g
    return ValidationResult(
        metric_name="mass_g",
        passed=passed,
        actual_value=mass,
        limit_value=max_mass_g,
        message=(
            f"estimated mass {mass:.1f} g "
            f"({'<=' if passed else '>'} limit {max_mass_g:.1f} g)"
        ),
    )


def check_bbox_within(
    mesh: trimesh.Trimesh, max_extent_mm: float
) -> ValidationResult:
    """Pass if no axis-aligned bbox dimension exceeds max_extent_mm."""
    extents = mesh.extents  # (dx, dy, dz)
    actual_max = float(np.max(extents))
    passed = actual_max <= max_extent_mm
    return ValidationResult(
        metric_name="bbox_max_extent_mm",
        passed=passed,
        actual_value=actual_max,
        limit_value=max_extent_mm,
        message=(
            f"max bbox extent {actual_max:.1f} mm "
            f"({'<=' if passed else '>'} limit {max_extent_mm:.1f} mm) "
            f"[extents = {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f}]"
        ),
    )
