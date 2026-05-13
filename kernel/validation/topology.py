"""Topology validators: manifold + watertight.

Both are hard invariants — every mesh leaving the kernel must pass these or
the kernel raises KernelError. See CLAUDE.md invariant #6.
"""

from __future__ import annotations

import trimesh
from pydantic import BaseModel, ConfigDict


class ValidationResult(BaseModel):
    """Structured outcome of a single validation check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str
    passed: bool
    actual_value: float
    limit_value: float
    message: str


def check_manifold(mesh: trimesh.Trimesh) -> ValidationResult:
    """Pass if the mesh is edge-manifold with consistent winding.

    trimesh's `is_winding_consistent` is the practical manifold check —
    a mesh with non-manifold edges cannot have a consistent winding. We
    also assert `is_volume`, which combines winding-consistency,
    watertightness, and absence of degenerate faces.
    """
    is_edge_manifold = bool(mesh.is_winding_consistent)
    is_volume = bool(mesh.is_volume)
    passed = is_edge_manifold and is_volume

    # Use numeric encoding for actual_value so the schema stays uniform across
    # all ValidationResults. 1.0 = pass, 0.0 = fail.
    return ValidationResult(
        metric_name="manifold",
        passed=passed,
        actual_value=1.0 if passed else 0.0,
        limit_value=1.0,
        message=(
            "mesh is manifold"
            if passed
            else (
                "mesh is non-manifold "
                f"(winding_consistent={is_edge_manifold}, is_volume={is_volume})"
            )
        ),
    )


def check_watertight(mesh: trimesh.Trimesh) -> ValidationResult:
    """Pass if trimesh reports the mesh as watertight."""
    passed = bool(mesh.is_watertight)
    return ValidationResult(
        metric_name="watertight",
        passed=passed,
        actual_value=1.0 if passed else 0.0,
        limit_value=1.0,
        message=(
            "mesh is watertight"
            if passed
            else f"mesh is not watertight (euler_number={mesh.euler_number})"
        ),
    )
