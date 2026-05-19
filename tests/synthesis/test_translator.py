"""Translator: correct op per type, magnitude sign, invariant #11, determinism."""

from __future__ import annotations

import pytest

from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.semantic import CorrectionZone, ReliefZone, StrapMount, TrimLine, VentRegion
from synthesis.resolve import ResolvedGeometry
from synthesis.translator import translate


def _p() -> Provenance:
    return Provenance(source=ProvenanceSource.LLM_PROPOSAL, rationale="fixture", confidence=0.8)


def _tv(v):
    return TracedValue(value=v, provenance=_p())


def _cz() -> CorrectionZone:
    return CorrectionZone(
        anatomical_anchors=["opisthocranion"],
        region_ref="opisthocranion",
        target_offset_mm=_tv(6.0),
        pressure_target_mmHg=_tv(12.0),
        schedule=_tv("progressive"),
        duration_weeks=_tv(12),
        transition_smoothing_mm=_tv(20.0),
        falloff_function=_tv("gaussian"),
    )


def _rz() -> ReliefZone:
    return ReliefZone(
        anatomical_anchors=["glabella"],
        region_ref="glabella",
        relief_offset_mm=_tv(4.0),
        reason=_tv("contralateral frontal bossing"),
        transition_smoothing_mm=_tv(18.0),
    )


def test_correction_zone_inward_offset_with_provenance() -> None:
    op = translate(
        _cz(), ResolvedGeometry(kind="correction_zone", center_xyz=(1, 2, 3), radius_mm=20.0)
    )[0]
    assert op.op_type == "offset_surface"
    assert op.inputs["direction"] == "inward"
    assert op.inputs["magnitude_mm"] == pytest.approx(6.0)
    assert op.inputs["signed_magnitude_mm"] == pytest.approx(-6.0)
    assert len(op.derived_from) == 1  # invariant #11


def test_relief_zone_outward_offset() -> None:
    rz = _rz()
    op = translate(rz, ResolvedGeometry(kind="relief_zone", center_xyz=(0, 0, 0), radius_mm=18.0))[
        0
    ]
    assert op.op_type == "offset_surface"
    assert op.inputs["direction"] == "outward"
    assert op.inputs["signed_magnitude_mm"] == pytest.approx(+4.0)
    assert op.derived_from == [rz.id]


def test_trim_vent_strap_op_types_and_provenance() -> None:
    tl = TrimLine(
        landmark_path=["a", "b", "c", "d"],
        height_offsets_mm=_tv([0.0, 0.0, 0.0, 0.0]),
        edge_treatment=_tv("rolled"),
        edge_thickness_mm=_tv(2.5),
    )
    top = translate(tl, ResolvedGeometry(kind="trim_line", path_points_xyz=[(0, 0, 0)] * 4))[0]
    assert top.op_type == "trim_cut" and top.derived_from == [tl.id]

    vent = VentRegion(
        anatomical_anchors=["vertex"],
        region_ref="vertex",
        pattern=_tv("hex_lattice"),
        open_area_fraction=_tv(0.18),
        feature_size_mm=_tv(6.0),
    )
    vop = translate(
        vent, ResolvedGeometry(kind="vent_region", center_xyz=(0, 0, 1), extent_mm=18.0)
    )[0]
    assert vop.op_type == "boolean_subtract" and vop.derived_from == [vent.id]

    strap = StrapMount(
        anatomical_anchors=["tragion_left"],
        position_ref="tragion_left",
        strap_width_mm=_tv(20.0),
        angle_deg=_tv(35.0),
        reinforcement_radius_mm=_tv(8.0),
    )
    sop = translate(strap, ResolvedGeometry(kind="strap_mount", anchor_xyz=(1, 0, -5)))[0]
    assert sop.op_type == "boolean_union" and sop.derived_from == [strap.id]


def test_every_op_has_non_empty_derived_from() -> None:
    """Invariant #11 across all five types."""
    cz, rz = _cz(), _rz()
    ops = []
    ops += translate(
        cz, ResolvedGeometry(kind="correction_zone", center_xyz=(0, 0, 0), radius_mm=10.0)
    )
    ops += translate(rz, ResolvedGeometry(kind="relief_zone", center_xyz=(0, 0, 0), radius_mm=10.0))
    for op in ops:
        assert op.derived_from, f"operation {op.op_type} has empty derived_from"


def test_translator_content_is_deterministic() -> None:
    cz = _cz()
    r = ResolvedGeometry(kind="correction_zone", center_xyz=(1.0, 2.0, 3.0), radius_mm=20.0)
    a, b = translate(cz, r)[0], translate(cz, r)[0]
    # op_id is uuid4-defaulted and excluded; content must be identical.
    assert (a.op_type, a.inputs, a.derived_from) == (b.op_type, b.inputs, b.derived_from)
