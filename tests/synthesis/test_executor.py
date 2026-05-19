"""Executor: phase grouping, loud failure with semantic id, constraint capture."""

from __future__ import annotations

import uuid

import pytest

from datamodel.operations import KernelOperation
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head
from synthesis.errors import ExecutionError
from synthesis.executor import execute

_PLAGIO = HeadConfig(
    occipital_flat_mm=15.0, occipital_flat_radius_mm=55.0, occipital_flat_lateral_offset_mm=35.0
)


def _offset_op(direction: str, sem_id: uuid.UUID) -> KernelOperation:
    return KernelOperation(
        op_type="offset_surface",
        inputs={
            "center_mm": [0.0, -75.0, 56.0],
            "radius_mm": 22.0,
            "magnitude_mm": 2.5,
            "signed_magnitude_mm": -2.5 if direction == "inward" else 2.5,
            "direction": direction,
            "falloff": "gaussian",
        },
        derived_from=[sem_id],
    )


def test_shell_phase_builds_valid_helmet_and_captures_constraints() -> None:
    head = generate_test_head(_PLAGIO)
    sem = uuid.uuid4()
    out = execute([_offset_op("inward", sem)], head)
    assert out.mesh.is_watertight and out.mesh.is_volume
    names = {c.name for c in out.constraints}
    assert {"manifold", "watertight", "min_wall_thickness_mm", "mass_g"} <= names
    # Audit references the shell phase and the semantic id.
    actions = [e.action for e in out.audit_events]
    assert "shell_phase" in actions and "validate" in actions
    shell_ev = next(e for e in out.audit_events if e.action == "shell_phase")
    assert str(sem) in shell_ev.payload["derived_from"]


def test_phase_order_offsets_then_trim_then_vent_then_strap() -> None:
    head = generate_test_head(_PLAGIO)
    sem = uuid.uuid4()
    trim = KernelOperation(
        op_type="trim_cut",
        inputs={
            "path_points_mm": [
                [0.0, 90.0, 40.0],
                [70.0, 0.0, 40.0],
                [0.0, -90.0, 40.0],
                [-70.0, 0.0, 40.0],
            ],
            "edge_treatment": "rolled",
            "edge_thickness_mm": 2.5,
        },
        derived_from=[sem],
    )
    out = execute([trim, _offset_op("inward", sem)], head)  # deliberately out of order
    actions = [e.action for e in out.audit_events if e.action.endswith("_phase")]
    # Shell is grouped/applied before trim regardless of input order.
    assert actions.index("shell_phase") < actions.index("trim_phase")
    assert out.mesh.is_watertight


def test_nonmanifold_phase_raises_with_semantic_id() -> None:
    head = generate_test_head(_PLAGIO)
    sem = uuid.uuid4()
    # A correction far deeper than the wall pierces the shell; build_shell
    # itself rejects degenerate input -> ExecutionError naming the sem id.
    bad = KernelOperation(
        op_type="offset_surface",
        inputs={
            "center_mm": [0.0, -75.0, 56.0],
            "radius_mm": 60.0,
            "magnitude_mm": 400.0,
            "signed_magnitude_mm": -400.0,
            "direction": "inward",
            "falloff": "gaussian",
        },
        derived_from=[sem],
    )
    with pytest.raises(ExecutionError) as ei:
        execute([bad], head)
    assert str(sem) in str(ei.value)


def test_more_than_one_trim_raises() -> None:
    head = generate_test_head(_PLAGIO)
    sem = uuid.uuid4()
    t = KernelOperation(
        op_type="trim_cut",
        inputs={
            "path_points_mm": [[0, 90, 40], [70, 0, 40], [0, -90, 40], [-70, 0, 40]],
            "edge_treatment": "straight",
            "edge_thickness_mm": 2.0,
        },
        derived_from=[sem],
    )
    with pytest.raises(ExecutionError, match="single trim_cut"):
        execute([_offset_op("inward", sem), t, t], head)
