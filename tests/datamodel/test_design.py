"""Design round-trip: one of each semantic subtype + ops + constraints."""

from __future__ import annotations

from datamodel.constraints import Constraint
from datamodel.design import Design
from datamodel.operations import KernelOperation
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    StrapMount,
    TrimLine,
    VentRegion,
)


def _p() -> Provenance:
    return Provenance(
        source=ProvenanceSource.TEMPLATE_DEFAULT,
        rationale="design fixture rationale",
        confidence=0.75,
    )


def _tv(v):
    return TracedValue(value=v, provenance=_p())


def _design() -> Design:
    return Design(
        version=2,
        semantic_objects=[
            CorrectionZone(
                anatomical_anchors=["opisthocranion"],
                region_ref="rpq",
                target_offset_mm=_tv(6.0),
                pressure_target_mmHg=_tv(12.0),
                schedule=_tv("progressive"),
                duration_weeks=_tv(12),
                transition_smoothing_mm=_tv(8.0),
                falloff_function=_tv("gaussian"),
            ),
            ReliefZone(
                anatomical_anchors=["vertex"],
                region_ref="crest",
                relief_offset_mm=_tv(3.0),
                reason=_tv("suture clearance"),
                transition_smoothing_mm=_tv(5.0),
            ),
            TrimLine(
                anatomical_anchors=["tragion_left", "tragion_right"],
                landmark_path=["glabella", "tragion_right", "opisthocranion"],
                height_offsets_mm=_tv([2.0, -1.5, -3.0]),
                edge_treatment=_tv("rolled"),
                edge_thickness_mm=_tv(2.5),
            ),
            VentRegion(
                anatomical_anchors=["vertex"],
                region_ref="dome",
                pattern=_tv("hex_lattice"),
                open_area_fraction=_tv(0.18),
                feature_size_mm=_tv(6.0),
            ),
            StrapMount(
                anatomical_anchors=["tragion_left"],
                position_ref="left_temporal",
                strap_width_mm=_tv(20.0),
                angle_deg=_tv(35.0),
                reinforcement_radius_mm=_tv(8.0),
            ),
        ],
        operations=[KernelOperation(op_type="offset_surface", inputs={"mm": 6.0})],
        output_artifacts={"helmet_stl": "sha256:" + "cd" * 32},
        constraints_checked=[
            Constraint(
                name="min_wall_thickness_mm",
                domain="manufacturing",
                expression="wall >= 3.0",
                satisfied=True,
                actual_value=3.5,
                limit_value=3.0,
                severity="error",
            )
        ],
        parent_version=1,
        change_summary="added vent + strap",
    )


def test_design_round_trips_to_exact_subclasses() -> None:
    d = _design()
    restored = Design.model_validate_json(d.model_dump_json())
    assert restored == d
    expected = [CorrectionZone, ReliefZone, TrimLine, VentRegion, StrapMount]
    for obj, cls in zip(restored.semantic_objects, expected, strict=False):
        assert type(obj) is cls


def test_design_serialization_is_deterministic() -> None:
    d = _design()
    assert d.model_dump_json() == d.model_dump_json()
    once = d.model_dump_json()
    assert once == Design.model_validate_json(once).model_dump_json()
