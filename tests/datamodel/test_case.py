"""Case aggregate: round-trip, determinism, and helper-method behavior."""

from __future__ import annotations

from anatomy.cleanup import CleanupReport
from anatomy.config import REGISTRATION_LANDMARKS
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import MeasurementResult
from datamodel.audit import AuditActor, AuditEvent
from datamodel.case import Case
from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.design import Design
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.scan_record import ScanRecord
from datamodel.semantic import CorrectionZone


def _prov(src: ProvenanceSource = ProvenanceSource.CLINICAL_INPUT) -> Provenance:
    return Provenance(source=src, rationale="hand-authored fixture rationale", confidence=0.9)


def _tv(value):
    return TracedValue(value=value, provenance=_prov())


def _landmark_set() -> LandmarkSet:
    lms = {
        n: Landmark(name=n, position_mm=(float(i), float(i) + 1, float(i) + 2), method="derived")
        for i, n in enumerate(REGISTRATION_LANDMARKS)
    }
    return LandmarkSet(frame="canonical", landmarks=lms)


def _scan_record() -> ScanRecord:
    return ScanRecord(
        mesh_hash="sha256:" + "ab" * 32,
        source_filename="synthetic_plagio.stl",
        landmark_set=_landmark_set(),
        measurements=[
            MeasurementResult(
                name="cephalic_index",
                value=80.4,
                unit="index",
                inputs_used=["euryon_left", "euryon_right"],
                quality="high",
                notes="",
            )
        ],
        cleanup_report=CleanupReport(
            was_already_manifold=True,
            duplicate_vertices_merged=0,
            degenerate_faces_removed=0,
            holes_filled=0,
            faces_removed=0,
            faces_added=0,
            normals_flipped=False,
        ),
        registered_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _clinical() -> CranialClinicalRecord:
    return CranialClinicalRecord(
        patient_age_months=_tv(7),
        provenance=_prov(),
        diagnosis=_tv(CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY),
        cvai=_tv(7.5),
        cephalic_index=_tv(80.4),
        severity=_tv("moderate"),
        laterality=_tv("right"),
        target_duration_weeks=_tv(12),
        asymmetry_pattern=_tv("right posterior flattening"),
        contraindications=[],
    )


def _design(version: int = 1) -> Design:
    return Design(
        version=version,
        semantic_objects=[
            CorrectionZone(
                anatomical_anchors=["opisthocranion"],
                region_ref="right_posterior",
                target_offset_mm=_tv(6.0),
                pressure_target_mmHg=_tv(12.0),
                schedule=_tv("progressive"),
                duration_weeks=_tv(12),
                transition_smoothing_mm=_tv(8.0),
                falloff_function=_tv("gaussian"),
            )
        ],
        change_summary="initial design",
    )


def _case() -> Case:
    return Case(
        device_class="cranial_helmet",
        clinical_narrative="7-month-old, right posterior flattening, referred for helmet.",
        clinical_record=_clinical(),
        scan_record=_scan_record(),
    )


def test_case_round_trips_lossless() -> None:
    case = _case()
    case.add_design_version(_design())
    case.append_audit(
        AuditEvent(actor=AuditActor.CLINICIAN, action="reviewed", rationale="looks good")
    )
    restored = Case.model_validate_json(case.model_dump_json())
    assert restored == case


def test_case_serialization_is_deterministic() -> None:
    case = _case()
    case.add_design_version(_design())
    assert case.model_dump_json() == case.model_dump_json()
    once = case.model_dump_json()
    twice = Case.model_validate_json(once).model_dump_json()
    assert once == twice


def test_add_design_version_sets_version_and_parent() -> None:
    case = _case()
    case.add_design_version(_design())
    case.add_design_version(_design())
    case.add_design_version(_design())
    versions = [d.version for d in case.design_versions]
    parents = [d.parent_version for d in case.design_versions]
    assert versions == [1, 2, 3]
    assert parents == [None, 1, 2]
    assert case.latest_design().version == 3
    # Each add_design_version appended a matching audit event.
    add_events = [e for e in case.audit_events if e.action == "add_design_version"]
    assert [e.payload["version"] for e in add_events] == [1, 2, 3]


def test_latest_design_none_when_empty() -> None:
    assert _case().latest_design() is None


def test_clinical_record_keeps_concrete_subtype() -> None:
    restored = Case.model_validate_json(_case().model_dump_json())
    assert isinstance(restored.clinical_record, CranialClinicalRecord)
    assert restored.clinical_record.diagnosis.value is CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY
