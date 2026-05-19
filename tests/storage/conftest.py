"""Storage-test fixtures: a fully-populated Case built from datamodel +
real Week-2 anatomy types (no kernel/scan pipeline needed for storage tests)."""

from __future__ import annotations

import pytest

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


def _p() -> Provenance:
    return Provenance(source=ProvenanceSource.CLINICAL_INPUT, rationale="fixture", confidence=0.9)


def _tv(v):
    return TracedValue(value=v, provenance=_p())


def _build_case() -> Case:
    lms = LandmarkSet(
        frame="canonical",
        landmarks={
            n: Landmark(name=n, position_mm=(float(i), float(i) + 1, 0.0), method="derived")
            for i, n in enumerate(REGISTRATION_LANDMARKS)
        },
    )
    scan = ScanRecord(
        mesh_hash="sha256:" + "ab" * 32,
        source_filename="s.stl",
        landmark_set=lms,
        measurements=[
            MeasurementResult(
                name="cvai",
                value=7.5,
                unit="percent",
                inputs_used=["x"],
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
    clinical = CranialClinicalRecord(
        patient_age_months=_tv(7),
        provenance=_p(),
        diagnosis=_tv(CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY),
        cvai=_tv(7.5),
        cephalic_index=_tv(80.4),
        severity=_tv("moderate"),
        laterality=_tv("right"),
        target_duration_weeks=_tv(12),
        asymmetry_pattern=_tv("right posterior"),
        contraindications=[],
    )
    case = Case(
        device_class="cranial_helmet",
        clinical_narrative="demo narrative",
        clinical_record=clinical,
        scan_record=scan,
    )
    case.add_design_version(
        Design(
            version=1,
            semantic_objects=[
                CorrectionZone(
                    region_ref="r",
                    target_offset_mm=_tv(6.0),
                    pressure_target_mmHg=_tv(12.0),
                    schedule=_tv("progressive"),
                    duration_weeks=_tv(12),
                    transition_smoothing_mm=_tv(8.0),
                    falloff_function=_tv("gaussian"),
                )
            ],
            change_summary="v1",
        )
    )
    case.append_audit(AuditEvent(actor=AuditActor.CLINICIAN, action="reviewed", rationale="ok"))
    return case


@pytest.fixture
def populated_case() -> Case:
    return _build_case()
