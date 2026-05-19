"""Phase-structured execution of a KernelOperation list -> helmet mesh.

NOT a generic op interpreter. Operations are grouped by phase, driven by
`op_type` (see the executor phase contract in CLAUDE.md):

  1. Shell  — ALL `offset_surface` ops together: inward corrections feed
              the Week 1 shell builder's per-region correction set; outward
              relief ops apply the additive Week-4 relief primitive.
  2. Trim   — the single `trim_cut` op via the Week 1 trim function
              (resolved xyz path -> azimuth/elevation control points).
  3. Vents  — each `boolean_subtract` op via the additive vent primitive,
              sequentially (the primitive canonicalizes internally).
  4. Straps — each `boolean_union` op via the additive strap primitive,
              sequentially (the primitive canonicalizes internally).

After all phases the Week 1 validators run and each result is captured as a
`datamodel.Constraint`. Non-manifold/empty geometry raises `ExecutionError`
naming the offending op and the semantic id it derived from (invariant #6).
The executor records constraints; it never loops back to fix them (that is
week 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from datamodel.audit import AuditActor, AuditEvent
from datamodel.constraints import Constraint
from datamodel.operations import KernelOperation
from kernel.config import CorrectionRegion, ShellConfig, TrimConfig, TrimControlPoint
from kernel.operations.shell import build_shell
from kernel.operations.trim import apply_trim
from kernel.primitives.relief import apply_relief
from kernel.primitives.straps import apply_strap_mount
from kernel.primitives.vents import apply_vent_pattern
from kernel.validation.geometry import check_bbox_within, check_mass, check_min_wall_thickness
from kernel.validation.topology import ValidationResult, check_manifold, check_watertight
from synthesis.errors import ExecutionError

# Week 1 validation defaults (ValidationConfig); kept here so the executor
# does not depend on a KernelConfig instance.
_MIN_WALL_MM = 3.0
_MAX_MASS_G = 250.0
_MAX_BBOX_MM = 250.0
_DENSITY_G_CM3 = 1.27
_TRIM_CUT_DEPTH_MM = 80.0

_DOMAIN = {
    "manifold": "manufacturing",
    "watertight": "manufacturing",
    "min_wall_thickness_mm": "manufacturing",
    "mass_g": "biomechanical",
    "bbox_max_extent_mm": "manufacturing",
}


@dataclass(frozen=True)
class ExecutionOutput:
    mesh: trimesh.Trimesh
    constraints: list[Constraint]
    audit_events: list[AuditEvent]


def _semantic_ids(ops: list[KernelOperation]) -> list[str]:
    out: list[str] = []
    for o in ops:
        out.extend(str(u) for u in o.derived_from)
    return out


def _audit(action: str, ops: list[KernelOperation], rationale: str) -> AuditEvent:
    return AuditEvent(
        actor=AuditActor.KERNEL,
        action=action,
        payload={
            "op_ids": [str(o.op_id) for o in ops],
            "op_types": [o.op_type for o in ops],
            "derived_from": _semantic_ids(ops),
        },
        rationale=rationale,
    )


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return the largest-volume connected component.

    Week 1's `apply_trim` leaves a tiny (~12 mm^3) disconnected Z-axis stub
    — an artifact of its 1 mm inner-radius cutter ring, flagged in the Week
    2/3 audits. It is not helmet geometry and breaks the strap primitive's
    single-solid fusion guard. Dropping it is deterministic post-processing
    (pick max |volume|, lexicographic tie-break on bounds), not a
    reimplementation of any Week 1 function.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    best = max(
        parts,
        key=lambda p: (abs(float(p.volume)), tuple(np.round(p.bounds.ravel(), 6))),
    )
    return best


def _gate(mesh: trimesh.Trimesh, op: KernelOperation, phase: str) -> None:
    if mesh is None or len(mesh.faces) == 0:
        raise ExecutionError(
            f"{phase} produced empty geometry "
            f"(op {op.op_id} {op.op_type}, derived_from={op.derived_from})"
        )
    if not (mesh.is_watertight and mesh.is_winding_consistent):
        raise ExecutionError(
            f"{phase} produced non-manifold/non-watertight geometry "
            f"(op {op.op_id} {op.op_type}, derived_from={op.derived_from})"
        )


# --------------------------------------------------------------------------- #
# Phase 1 — shell (corrections + relief, fed together)                         #
# --------------------------------------------------------------------------- #


def _shell_phase(
    head_mesh: trimesh.Trimesh,
    offset_ops: list[KernelOperation],
    events: list[AuditEvent],
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    inward = [o for o in offset_ops if o.inputs.get("direction") == "inward"]
    outward = [o for o in offset_ops if o.inputs.get("direction") == "outward"]

    regions = [
        CorrectionRegion(
            x_mm=float(o.inputs["center_mm"][0]),
            y_mm=float(o.inputs["center_mm"][1]),
            z_mm=float(o.inputs["center_mm"][2]),
            radius_mm=float(o.inputs["radius_mm"]),
            magnitude_mm=float(o.inputs["magnitude_mm"]),
            falloff=str(o.inputs["falloff"]),
        )
        for o in inward
    ]
    cfg = ShellConfig(relief_mm=3.0, wall_thickness_mm=4.0, correction_regions=regions)
    try:
        shell, inner, outer = build_shell(head_mesh, cfg)
    except Exception as exc:
        off = inward[0] if inward else (offset_ops[0] if offset_ops else None)
        raise ExecutionError(
            f"shell phase build_shell failed: {exc} "
            f"(derived_from={off.derived_from if off else []})"
        ) from exc

    # Outward relief: additive Week-4 primitive (Week 1 shell builder has no
    # outward knob). Applied sequentially onto the built shell.
    for o in outward:
        try:
            shell = apply_relief(
                shell,
                center_mm=tuple(float(c) for c in o.inputs["center_mm"]),
                radius_mm=float(o.inputs["radius_mm"]),
                depth_mm=float(o.inputs["magnitude_mm"]),
                falloff=str(o.inputs["falloff"]),
            )
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(
                f"shell phase relief failed: {exc} "
                f"(op {o.op_id}, derived_from={o.derived_from})"
            ) from exc
        _gate(shell, o, "shell phase (relief)")

    if offset_ops:
        events.append(
            _audit(
                "shell_phase",
                offset_ops,
                f"built corrected shell from {len(inward)} inward correction(s) "
                f"+ {len(outward)} relief region(s)",
            )
        )
    return shell, inner, outer


# --------------------------------------------------------------------------- #
# Phase 2 — trim                                                              #
# --------------------------------------------------------------------------- #


def _xyz_to_control_point(
    p: list[float], centroid: np.ndarray, bbox_radius: float
) -> TrimControlPoint:
    dx = float(p[0]) - float(centroid[0])
    dy = float(p[1]) - float(centroid[1])
    # Week 1 convention: azimuth 0 = +Y, +90 = +X  => atan2(x, y).
    az = math.degrees(math.atan2(dx, dy)) % 360.0
    sin_el = (float(p[2]) - float(centroid[2])) / bbox_radius if bbox_radius > 0 else 0.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    return TrimControlPoint(azimuth_deg=az, elevation_deg=el)


def _trim_phase(
    mesh: trimesh.Trimesh,
    trim_ops: list[KernelOperation],
    events: list[AuditEvent],
) -> trimesh.Trimesh:
    if not trim_ops:
        return mesh
    if len(trim_ops) > 1:
        raise ExecutionError(
            f"trim phase expects a single trim_cut op, got {len(trim_ops)} "
            f"(derived_from={[str(u) for o in trim_ops for u in o.derived_from]})"
        )
    op = trim_ops[0]
    centroid = mesh.bounds.mean(axis=0)
    bbox_radius = float(0.5 * np.linalg.norm(mesh.extents))
    pts = op.inputs["path_points_mm"]
    cps = [_xyz_to_control_point(p, centroid, bbox_radius) for p in pts]
    try:
        cfg = TrimConfig(control_points=cps, cut_depth_mm=_TRIM_CUT_DEPTH_MM)
    except Exception as exc:
        raise ExecutionError(
            f"trim phase: resolved path does not form a valid trim curve "
            f"({exc}) (op {op.op_id}, derived_from={op.derived_from})"
        ) from exc
    try:
        trimmed = apply_trim(mesh, cfg)
    except Exception as exc:
        raise ExecutionError(
            f"trim phase apply_trim failed: {exc} "
            f"(op {op.op_id}, derived_from={op.derived_from})"
        ) from exc
    # Drop the known Week-1 trim stub artifact so downstream vent/strap
    # booleans operate on a single clean solid.
    trimmed = _largest_component(trimmed)
    _gate(trimmed, op, "trim phase")
    events.append(_audit("trim_phase", [op], "applied trim cut from resolved landmark path"))
    return trimmed


# --------------------------------------------------------------------------- #
# Phases 3 & 4 — vents then straps (sequential booleans)                       #
# --------------------------------------------------------------------------- #


def _vent_phase(
    mesh: trimesh.Trimesh, vent_ops: list[KernelOperation], events: list[AuditEvent]
) -> trimesh.Trimesh:
    for op in vent_ops:
        try:
            mesh = apply_vent_pattern(
                mesh,
                center_mm=tuple(float(c) for c in op.inputs["center_mm"]),
                extent_mm=float(op.inputs["extent_mm"]),
                pattern=str(op.inputs["pattern"]),
                open_area_fraction=float(op.inputs["open_area_fraction"]),
                feature_size_mm=float(op.inputs["feature_size_mm"]),
            )
        except Exception as exc:
            raise ExecutionError(
                f"vent phase failed: {exc} (op {op.op_id}, derived_from={op.derived_from})"
            ) from exc
        _gate(mesh, op, "vent phase")
        events.append(_audit("vent_boolean", [op], "subtracted patterned vent cutter"))
    if vent_ops:
        events.append(_audit("vent_phase", vent_ops, f"applied {len(vent_ops)} vent region(s)"))
    return mesh


def _strap_phase(
    mesh: trimesh.Trimesh, strap_ops: list[KernelOperation], events: list[AuditEvent]
) -> trimesh.Trimesh:
    for op in strap_ops:
        # The strap mounts on the actual trimmed helmet. Snap the semantic
        # anchor (a landmark position) onto the current helmet surface so
        # the tab fuses to the real wall / trim edge.
        raw = np.asarray(op.inputs["anchor_mm"], dtype=np.float64).reshape(1, 3)
        closest, _d, _f = trimesh.proximity.closest_point(mesh, raw)
        anchor_on_helmet = tuple(float(c) for c in closest[0])
        try:
            mesh = apply_strap_mount(
                mesh,
                anchor_mm=anchor_on_helmet,
                strap_width_mm=float(op.inputs["strap_width_mm"]),
                angle_deg=float(op.inputs["angle_deg"]),
                reinforcement_radius_mm=float(op.inputs["reinforcement_radius_mm"]),
            )
        except Exception as exc:
            raise ExecutionError(
                f"strap phase failed: {exc} (op {op.op_id}, derived_from={op.derived_from})"
            ) from exc
        _gate(mesh, op, "strap phase")
        events.append(_audit("strap_boolean", [op], "unioned lofted strap-mount tab"))
    if strap_ops:
        events.append(_audit("strap_phase", strap_ops, f"applied {len(strap_ops)} strap mount(s)"))
    return mesh


# --------------------------------------------------------------------------- #
# Validation -> Constraints                                                   #
# --------------------------------------------------------------------------- #


def _to_constraint(r: ValidationResult) -> Constraint:
    return Constraint(
        name=r.metric_name,
        domain=_DOMAIN.get(r.metric_name, "manufacturing"),
        expression=r.message,
        satisfied=bool(r.passed),
        actual_value=float(r.actual_value),
        limit_value=float(r.limit_value),
        severity="error",
    )


def _capture_constraints(
    helmet: trimesh.Trimesh,
    inner: trimesh.Trimesh,
    outer: trimesh.Trimesh,
) -> list[Constraint]:
    # Wall thickness is a property of the offset construction (Week 1's own
    # generate.py measures it on the untrimmed inner/outer for the same
    # reason: trim/vent/strap do not change the bulk wall).
    results = [
        check_manifold(helmet),
        check_watertight(helmet),
        check_min_wall_thickness(inner, outer, _MIN_WALL_MM),
        check_mass(helmet, _DENSITY_G_CM3, _MAX_MASS_G),
        check_bbox_within(helmet, _MAX_BBOX_MM),
    ]
    return [_to_constraint(r) for r in results]


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def execute(operations: list[KernelOperation], head_mesh: trimesh.Trimesh) -> ExecutionOutput:
    """Run the operation list phase-structured against the patient head mesh.

    Returns the helmet mesh, the captured constraints, and the ordered audit
    events (one per phase, plus one per individual vent/strap boolean).
    """
    events: list[AuditEvent] = []
    offset_ops = [o for o in operations if o.op_type == "offset_surface"]
    trim_ops = [o for o in operations if o.op_type == "trim_cut"]
    vent_ops = [o for o in operations if o.op_type == "boolean_subtract"]
    strap_ops = [o for o in operations if o.op_type == "boolean_union"]

    shell, inner, outer = _shell_phase(head_mesh, offset_ops, events)
    mesh = _trim_phase(shell, trim_ops, events)
    mesh = _vent_phase(mesh, vent_ops, events)
    mesh = _strap_phase(mesh, strap_ops, events)

    constraints = _capture_constraints(mesh, inner, outer)
    events.append(
        AuditEvent(
            actor=AuditActor.VALIDATOR,
            action="validate",
            payload={
                "constraints": [{"name": c.name, "satisfied": c.satisfied} for c in constraints]
            },
            rationale="Week 1 validators run on the synthesized helmet",
        )
    )
    return ExecutionOutput(mesh=mesh, constraints=constraints, audit_events=events)
