"""Validation primitives — topology and geometry checks."""

from kernel.validation.geometry import (
    check_bbox_within,
    check_min_wall_thickness,
    estimate_mass_g,
)
from kernel.validation.topology import (
    ValidationResult,
    check_manifold,
    check_watertight,
)

__all__ = [
    "ValidationResult",
    "check_manifold",
    "check_watertight",
    "check_min_wall_thickness",
    "check_bbox_within",
    "estimate_mass_g",
]
