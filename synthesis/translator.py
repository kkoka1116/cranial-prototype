"""Semantic object + resolved geometry -> deterministic KernelOperation list.

The translator is pure: same inputs -> identical operation content. (op_id
defaults to uuid4, so determinism comparisons exclude op_id and compare
op_type/inputs/derived_from — documented here and asserted in tests.)

Every emitted operation has a non-empty `derived_from` (invariant #11). The
operation list is the auditable record of what geometry will be produced
and why; the executor is the deterministic geometry producer.
"""

from __future__ import annotations

from datamodel.operations import KernelOperation
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    SemanticObject,
    StrapMount,
    TrimLine,
    VentRegion,
)
from synthesis.errors import SynthesisError
from synthesis.resolve import ResolvedGeometry


def _offset_op(
    obj: CorrectionZone | ReliefZone,
    resolved: ResolvedGeometry,
    *,
    signed_magnitude_mm: float,
    direction: str,
) -> KernelOperation:
    if resolved.center_xyz is None or resolved.radius_mm is None:
        raise SynthesisError(
            f"{type(obj).__name__} {obj.id}: resolved geometry lacks center/radius"
        )
    return KernelOperation(
        op_type="offset_surface",
        inputs={
            "center_mm": list(resolved.center_xyz),
            "radius_mm": resolved.radius_mm,
            # Week 1 CorrectionRegion.magnitude_mm is gt=0 and means INWARD
            # depth. We carry the kernel-facing positive magnitude plus the
            # signed clinical intent + direction for the audit record.
            "magnitude_mm": abs(signed_magnitude_mm),
            "signed_magnitude_mm": signed_magnitude_mm,
            "direction": direction,
            "falloff": obj.falloff_function.value
            if isinstance(obj, CorrectionZone)
            else "gaussian",
        },
        derived_from=[obj.id],
    )


def translate(obj: SemanticObject, resolved: ResolvedGeometry) -> list[KernelOperation]:
    """Translate one semantic object into its KernelOperation(s).

    Implemented this slice: CorrectionZone (inward), ReliefZone (outward).
    TrimLine/VentRegion/StrapMount are added in the full translator stage.
    """
    if isinstance(obj, CorrectionZone):
        return [
            _offset_op(
                obj,
                resolved,
                signed_magnitude_mm=-float(obj.target_offset_mm.value),
                direction="inward",
            )
        ]
    if isinstance(obj, ReliefZone):
        return [
            _offset_op(
                obj,
                resolved,
                signed_magnitude_mm=+float(obj.relief_offset_mm.value),
                direction="outward",
            )
        ]
    if isinstance(obj, TrimLine):
        if not resolved.path_points_xyz:
            raise SynthesisError(f"TrimLine {obj.id}: resolved geometry lacks path_points_xyz")
        return [
            KernelOperation(
                op_type="trim_cut",
                inputs={
                    "path_points_mm": [list(p) for p in resolved.path_points_xyz],
                    "edge_treatment": obj.edge_treatment.value,
                    "edge_thickness_mm": float(obj.edge_thickness_mm.value),
                },
                derived_from=[obj.id],
            )
        ]
    if isinstance(obj, VentRegion):
        if resolved.center_xyz is None or resolved.extent_mm is None:
            raise SynthesisError(f"VentRegion {obj.id}: resolved geometry lacks center/extent")
        return [
            KernelOperation(
                op_type="boolean_subtract",
                inputs={
                    "center_mm": list(resolved.center_xyz),
                    "extent_mm": resolved.extent_mm,
                    "pattern": obj.pattern.value,
                    "open_area_fraction": float(obj.open_area_fraction.value),
                    "feature_size_mm": float(obj.feature_size_mm.value),
                },
                derived_from=[obj.id],
            )
        ]
    if isinstance(obj, StrapMount):
        if resolved.anchor_xyz is None:
            raise SynthesisError(f"StrapMount {obj.id}: resolved geometry lacks anchor_xyz")
        return [
            KernelOperation(
                op_type="boolean_union",
                inputs={
                    "anchor_mm": list(resolved.anchor_xyz),
                    "strap_width_mm": float(obj.strap_width_mm.value),
                    "angle_deg": float(obj.angle_deg.value),
                    "reinforcement_radius_mm": float(obj.reinforcement_radius_mm.value),
                },
                derived_from=[obj.id],
            )
        ]
    raise SynthesisError(f"translate does not handle {type(obj).__name__} (object {obj.id})")
