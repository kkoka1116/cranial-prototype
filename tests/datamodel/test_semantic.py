"""Discriminated-union round-trip — the highest-risk test in week 3.

A list containing one of each of the five semantic subtypes must serialize
and deserialize back to the EXACT same five typed subclasses, and repeated
serialization must be byte-identical.
"""

from __future__ import annotations

from pydantic import TypeAdapter

from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    SemanticObjectUnion,
    StrapMount,
    TrimLine,
    VentRegion,
)


def _prov() -> Provenance:
    return Provenance(
        source=ProvenanceSource.LLM_PROPOSAL,
        rationale="hand-authored test fixture standing in for a week-6 proposal",
        confidence=0.8,
    )


def _tv(value):
    return TracedValue(value=value, provenance=_prov())


def _one_of_each() -> list:
    return [
        CorrectionZone(
            anatomical_anchors=["opisthocranion", "euryon_right"],
            region_ref="right_posterior_quadrant",
            target_offset_mm=_tv(6.0),
            pressure_target_mmHg=_tv(12.0),
            schedule=_tv("progressive"),
            duration_weeks=_tv(12),
            transition_smoothing_mm=_tv(8.0),
            falloff_function=_tv("gaussian"),
        ),
        ReliefZone(
            anatomical_anchors=["vertex"],
            region_ref="sagittal_crest",
            relief_offset_mm=_tv(3.0),
            reason=_tv("avoid pressure over the sagittal suture"),
            transition_smoothing_mm=_tv(5.0),
        ),
        TrimLine(
            anatomical_anchors=["tragion_left", "tragion_right"],
            landmark_path=["glabella", "tragion_right", "opisthocranion", "tragion_left"],
            height_offsets_mm=_tv([2.0, -1.5, 0.0, -1.5]),
            edge_treatment=_tv("rolled"),
            edge_thickness_mm=_tv(2.5),
        ),
        VentRegion(
            anatomical_anchors=["vertex"],
            region_ref="superior_dome",
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
    ]


def test_discriminated_union_round_trips_to_exact_subclasses() -> None:
    adapter = TypeAdapter(list[SemanticObjectUnion])
    original = _one_of_each()

    dumped = adapter.dump_json(original)
    restored = adapter.validate_json(dumped)

    assert len(restored) == 5
    expected_types = [CorrectionZone, ReliefZone, TrimLine, VentRegion, StrapMount]
    for obj, exp_cls in zip(restored, expected_types, strict=False):
        assert type(obj) is exp_cls, f"expected {exp_cls.__name__}, got {type(obj).__name__}"
    # Deep equality, field by field.
    assert restored == original


def test_discriminated_union_serialization_is_deterministic() -> None:
    adapter = TypeAdapter(list[SemanticObjectUnion])
    objs = _one_of_each()
    assert adapter.dump_json(objs) == adapter.dump_json(objs)
    # And stable across a round-trip.
    once = adapter.dump_json(objs)
    twice = adapter.dump_json(adapter.validate_json(once))
    assert once == twice


def test_traced_parameters_survive_union_round_trip() -> None:
    adapter = TypeAdapter(list[SemanticObjectUnion])
    restored = adapter.validate_json(adapter.dump_json(_one_of_each()))
    cz = restored[0]
    assert isinstance(cz, CorrectionZone)
    assert cz.target_offset_mm.value == 6.0
    assert cz.target_offset_mm.provenance.source is ProvenanceSource.LLM_PROPOSAL
    assert cz.target_offset_mm.provenance.confidence == 0.8
    tl = restored[2]
    assert isinstance(tl, TrimLine)
    assert tl.height_offsets_mm.value == [2.0, -1.5, 0.0, -1.5]


def test_wrong_discriminator_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    adapter = TypeAdapter(list[SemanticObjectUnion])
    with pytest.raises(ValidationError):
        adapter.validate_python([{"type": "not_a_real_type", "region_ref": "x"}])
