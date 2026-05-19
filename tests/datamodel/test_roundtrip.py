"""Centerpiece: lossless + deterministic round-trip for every model.

Each case: build a populated instance, assert
model_validate_json(model_dump_json()) is deep-equal, and that repeated /
post-round-trip serialization is byte-identical. (Case is covered in
test_case.py; the discriminated union in test_semantic.py.)
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, TypeAdapter

from anatomy.cleanup import CleanupReport
from anatomy.config import REGISTRATION_LANDMARKS
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import MeasurementResult
from datamodel.audit import AuditActor, AuditEvent
from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.constraints import Constraint
from datamodel.operations import KernelOperation
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.scan_record import ScanRecord
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    StrapMount,
    TrimLine,
    VentRegion,
)


def _p() -> Provenance:
    return Provenance(
        source=ProvenanceSource.LLM_PROPOSAL,
        rationale="round-trip fixture rationale",
        agent_id="agent-1",
        prior_value={"k": 1.0},
        confidence=0.5,
    )


def _tv(v):
    return TracedValue(value=v, provenance=_p())


def _landmarks() -> LandmarkSet:
    return LandmarkSet(
        frame="canonical",
        landmarks={
            n: Landmark(
                name=n, position_mm=(float(i), float(i) + 1, float(i) + 2), method="derived"
            )
            for i, n in enumerate(REGISTRATION_LANDMARKS)
        },
    )


def _scan_record() -> ScanRecord:
    return ScanRecord(
        mesh_hash="sha256:" + "ef" * 32,
        source_filename="s.stl",
        landmark_set=_landmarks(),
        measurements=[
            MeasurementResult(
                name="cvai", value=7.5, unit="percent", inputs_used=["x"], quality="high", notes="n"
            )
        ],
        cleanup_report=CleanupReport(
            was_already_manifold=True,
            duplicate_vertices_merged=1,
            degenerate_faces_removed=2,
            holes_filled=3,
            faces_removed=4,
            faces_added=5,
            normals_flipped=True,
            notes="r",
        ),
        registered_transform=[
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _cranial() -> CranialClinicalRecord:
    return CranialClinicalRecord(
        patient_age_months=_tv(7),
        notes="n",
        provenance=_p(),
        diagnosis=_tv(CranialDiagnosis.BRACHYCEPHALY),
        cvai=_tv(2.1),
        cephalic_index=_tv(92.0),
        severity=_tv("severe"),
        laterality=_tv(None),
        target_duration_weeks=_tv(16),
        asymmetry_pattern=_tv(None),
        contraindications=["scalp_lesion"],
    )


_INSTANCES = {
    "Provenance": _p(),
    "AuditEvent": AuditEvent(
        actor=AuditActor.VALIDATOR, action="checked", payload={"a": 1}, rationale="r"
    ),
    "Constraint": Constraint(
        name="c",
        domain="regulatory",
        expression="x<=1",
        satisfied=False,
        actual_value=2.0,
        limit_value=1.0,
        severity="warning",
    ),
    "KernelOperation": KernelOperation(op_type="trim_cut", inputs={"z": 1.0}),
    "ScanRecord": _scan_record(),
    "CranialClinicalRecord": _cranial(),
    "CorrectionZone": CorrectionZone(
        region_ref="r",
        target_offset_mm=_tv(6.0),
        pressure_target_mmHg=_tv(12.0),
        schedule=_tv("constant"),
        duration_weeks=_tv(12),
        transition_smoothing_mm=_tv(8.0),
        falloff_function=_tv("linear"),
    ),
    "ReliefZone": ReliefZone(
        region_ref="r",
        relief_offset_mm=_tv(3.0),
        reason=_tv("why"),
        transition_smoothing_mm=_tv(5.0),
    ),
    "TrimLine": TrimLine(
        landmark_path=["a", "b"],
        height_offsets_mm=_tv([1.0, 2.0]),
        edge_treatment=_tv("flared"),
        edge_thickness_mm=_tv(2.0),
    ),
    "VentRegion": VentRegion(
        region_ref="r",
        pattern=_tv("slot"),
        open_area_fraction=_tv(0.2),
        feature_size_mm=_tv(6.0),
    ),
    "StrapMount": StrapMount(
        position_ref="r",
        strap_width_mm=_tv(20.0),
        angle_deg=_tv(30.0),
        reinforcement_radius_mm=_tv(8.0),
    ),
}


@pytest.mark.parametrize("name", list(_INSTANCES))
def test_model_round_trips_lossless_and_deterministic(name: str) -> None:
    obj = _INSTANCES[name]
    cls = type(obj)
    restored = cls.model_validate_json(obj.model_dump_json())
    assert restored == obj, f"{name} lost information on round-trip"
    once = obj.model_dump_json()
    assert once == obj.model_dump_json(), f"{name} non-deterministic on repeat dump"
    assert (
        once == cls.model_validate_json(once).model_dump_json()
    ), f"{name} non-deterministic across round-trip"


@pytest.mark.parametrize(
    ("tp", "value"),
    [
        (float, 1.5),
        (int, 7),
        (str, "hello"),
        (bool, True),
        (list[float], [1.0, 2.5, -3.0]),
    ],
)
def test_traced_value_round_trips_over_supported_types(tp, value) -> None:
    class _Holder(BaseModel):
        tv: TracedValue[tp]  # type: ignore[valid-type]

    h = _Holder(tv=TracedValue(value=value, provenance=_p()))
    restored = _Holder.model_validate_json(h.model_dump_json())
    assert restored == h
    once = h.model_dump_json()
    assert once == _Holder.model_validate_json(once).model_dump_json()


def test_traced_value_literal_round_trips() -> None:
    ad = TypeAdapter(CorrectionZone)
    cz = _INSTANCES["CorrectionZone"]
    restored = ad.validate_json(ad.dump_json(cz))
    assert restored.schedule.value == "constant"
    assert restored.falloff_function.value == "linear"
    assert restored == cz
