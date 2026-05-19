"""Week-3 integration proof: assemble a full Case end to end, no LLM.

Real Week-1 + Week-2 outputs + hand-authored clinical/design data ->
persisted Case -> reloaded -> deep-equality assertion (the week's
acceptance gate) -> committed examples/demo_case.json + a one-screen
summary.

Run: uv run python scripts/build_demo_case.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from anatomy.cleanup import clean_scan
from anatomy.config import LANDMARK_ORDER
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import all_measurements
from anatomy.registration import apply_transform, register
from anatomy.synthetic import derive_landmarks_from_synthetic
from datamodel.audit import AuditActor, AuditEvent
from datamodel.case import Case
from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.constraints import Constraint
from datamodel.design import Design
from datamodel.operations import KernelOperation
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.scan_record import ScanRecord
from datamodel.semantic import CorrectionZone, ReliefZone, TrimLine
from kernel.config import HeadConfig, KernelConfig
from kernel.generate import run_pipeline
from kernel.primitives.test_head import generate_test_head
from storage.repository import CaseRepository

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# A plagiocephaly case mirroring the Week-1 moderate_right_plagio example.
_HEAD = HeadConfig(
    occipital_flat_mm=15.0,
    occipital_flat_radius_mm=55.0,
    occipital_flat_lateral_offset_mm=35.0,
)


def _prov(source: ProvenanceSource, rationale: str, confidence: float | None = None) -> Provenance:
    return Provenance(source=source, rationale=rationale, confidence=confidence)


def _clinician_tv(value, rationale: str):
    return TracedValue(
        value=value,
        provenance=_prov(ProvenanceSource.CLINICAL_INPUT, rationale, confidence=0.95),
    )


def _build_scan_record(repo: CaseRepository) -> tuple[ScanRecord, str]:
    """Real Week-2 pipeline: synthetic head -> derive -> clean -> register
    -> measure. Returns the ScanRecord and the registered-mesh blob hash."""
    derived = derive_landmarks_from_synthetic(_HEAD)
    # Treat the derived (canonical) landmarks as a freshly-picked "source"
    # set so the real registration path runs (identity-ish recovery).
    source = LandmarkSet(
        frame="source",
        landmarks={
            n: Landmark(
                name=n,
                position_mm=derived.landmarks[n].position_mm,
                method="manual",
            )
            for n in LANDMARK_ORDER
        },
    )
    raw_mesh = generate_test_head(_HEAD)
    cleaned = clean_scan(raw_mesh)
    transform, canonical_landmarks = register(source)
    registered_mesh = apply_transform(cleaned.mesh, transform)
    measurements = list(all_measurements(registered_mesh, canonical_landmarks).values())

    stl_bytes = registered_mesh.export(file_type="stl")
    mesh_hash = repo.save_blob(stl_bytes, "model/stl")

    scan = ScanRecord(
        mesh_hash=mesh_hash,
        source_filename="synthetic_moderate_right_plagio.stl",
        landmark_set=canonical_landmarks,
        measurements=measurements,
        cleanup_report=cleaned.report,
        registered_transform=[[float(x) for x in row] for row in transform.tolist()],
    )
    return scan, mesh_hash


def _build_clinical(measurements: list) -> CranialClinicalRecord:
    by_name = {m.name: m for m in measurements}
    cvai_val = round(by_name["cvai"].value, 2)
    ci_val = round(by_name["cephalic_index"].value, 2)
    return CranialClinicalRecord(
        patient_age_months=_clinician_tv(
            7, "age at intake from referral form; drives expected treatment duration"
        ),
        provenance=_prov(
            ProvenanceSource.CLINICAL_INPUT,
            "clinical record authored from intake exam and the registered scan",
        ),
        diagnosis=_clinician_tv(
            CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY,
            "right posterior flattening with contralateral frontal bossing on exam",
        ),
        cvai=TracedValue(
            value=cvai_val,
            provenance=_prov(
                ProvenanceSource.KERNEL_DERIVATION,
                f"CVAI measured from the registered scan ({cvai_val}%)",
                confidence=0.9,
            ),
        ),
        cephalic_index=TracedValue(
            value=ci_val,
            provenance=_prov(
                ProvenanceSource.KERNEL_DERIVATION,
                f"cephalic index measured from the registered scan ({ci_val})",
                confidence=0.9,
            ),
        ),
        severity=_clinician_tv(
            "moderate", "CVAI in the 7-9 band is clinically moderate plagiocephaly"
        ),
        laterality=_clinician_tv("right", "flattening is right-posterior"),
        target_duration_weeks=_clinician_tv(
            12, "typical course for moderate plagiocephaly at this age"
        ),
        asymmetry_pattern=_clinician_tv(
            "right occipital flattening, left frontal bossing",
            "documented asymmetry pattern from exam",
        ),
        contraindications=[],
    )


def _design_tv(value, rationale: str):
    return TracedValue(
        value=value,
        provenance=_prov(ProvenanceSource.TEMPLATE_DEFAULT, rationale, confidence=0.8),
    )


def _build_design() -> Design:
    correction = CorrectionZone(
        anatomical_anchors=["opisthocranion", "euryon_right"],
        region_ref="right_posterior_quadrant",
        target_offset_mm=_design_tv(6.0, "offset sized to the measured right-posterior deficit"),
        pressure_target_mmHg=_design_tv(12.0, "gentle remodeling pressure for an infant"),
        schedule=_design_tv("progressive", "ramp pressure to improve tolerance"),
        duration_weeks=_design_tv(12, "matches the clinical target duration"),
        transition_smoothing_mm=_design_tv(8.0, "smooth blend to avoid a pressure ridge"),
        falloff_function=_design_tv("gaussian", "smooth radial falloff"),
    )
    relief = ReliefZone(
        anatomical_anchors=["vertex"],
        region_ref="sagittal_crest",
        relief_offset_mm=_design_tv(3.0, "clearance over the sagittal suture"),
        reason=_design_tv("avoid pressure over the suture line", "standard relief practice"),
        transition_smoothing_mm=_design_tv(5.0, "smooth blend out of the relief"),
    )
    trim = TrimLine(
        anatomical_anchors=["tragion_left", "tragion_right"],
        landmark_path=["glabella", "tragion_right", "opisthocranion", "tragion_left"],
        height_offsets_mm=_design_tv(
            [2.0, -1.5, -3.0, -1.5], "trim clears the ears and brow per template"
        ),
        edge_treatment=_design_tv("rolled", "rolled edge for skin comfort"),
        edge_thickness_mm=_design_tv(2.5, "manufacturable edge thickness"),
    )
    return Design(
        version=1,
        semantic_objects=[correction, relief, trim],
        change_summary="initial hand-authored design (stands in for week-6 LLM proposal)",
    )


def _run_kernel_and_record(repo: CaseRepository) -> tuple[str, list, list]:
    """Run the Week-1 kernel directly; return (helmet blob hash,
    KernelOperations, Constraints recorded from the validators)."""
    cfg = KernelConfig.from_yaml(EXAMPLES / "moderate_right_plagio.yaml")
    helmet, validation_results = run_pipeline(cfg)
    helmet_hash = repo.save_blob(helmet.export(file_type="stl"), "model/stl")

    operations = [
        KernelOperation(op_type="generate_test_head", inputs={"config": "moderate_right_plagio"}),
        KernelOperation(op_type="build_shell", inputs={"wall_thickness_mm": 4.5}),
        KernelOperation(op_type="apply_trim", inputs={"trim": "moderate_right_plagio"}),
    ]
    constraints = [
        Constraint(
            name=r.metric_name,
            domain="manufacturing" if "wall" in r.metric_name else "biomechanical",
            expression=r.message,
            satisfied=bool(r.passed),
            actual_value=float(r.actual_value),
            limit_value=float(r.limit_value),
            severity="error",
        )
        for r in validation_results
    ]
    return helmet_hash, operations, constraints


def build_demo_case(repo: CaseRepository) -> Case:
    scan, _mesh_hash = _build_scan_record(repo)
    clinical = _build_clinical(scan.measurements)
    helmet_hash, operations, constraints = _run_kernel_and_record(repo)

    design = _build_design()
    design = design.model_copy(
        update={
            "operations": operations,
            "constraints_checked": constraints,
            "output_artifacts": {"helmet_stl": helmet_hash, "registered_scan_stl": scan.mesh_hash},
        }
    )

    case = Case(
        device_class="cranial_helmet",
        clinical_narrative=(
            "7-month-old presenting with right posterior positional plagiocephaly and "
            "mild contralateral frontal bossing. Referred for a cranial remolding helmet. "
            "No torticollis. Parents counseled on repositioning and helmet therapy."
        ),
        clinical_record=clinical,
        scan_record=scan,
    )
    case.append_audit(
        AuditEvent(
            actor=AuditActor.SYSTEM,
            action="scan_ingested",
            target_id=case.case_id,
            rationale="synthetic plagiocephaly scan cleaned, registered, measured",
        )
    )
    case.append_audit(
        AuditEvent(
            actor=AuditActor.CLINICIAN,
            action="clinical_record_authored",
            target_id=case.case_id,
            rationale="diagnosis and targets entered from intake exam",
        )
    )
    case.add_design_version(design)
    case.append_audit(
        AuditEvent(
            actor=AuditActor.KERNEL,
            action="helmet_generated",
            target_id=case.case_id,
            payload={"helmet_stl": helmet_hash},
            rationale="Week-1 kernel produced a manifold + watertight helmet",
        )
    )
    return case


def _summary(case: Case) -> str:
    cr = case.clinical_record
    latest = case.latest_design()
    lines = [
        "=" * 68,
        f"CASE {case.case_id}",
        "=" * 68,
        f"  device_class     : {case.device_class}",
        f"  diagnosis        : {cr.diagnosis.value.value}  "
        f"(severity {cr.severity.value}, {cr.laterality.value})",
        f"  CVAI / CI        : {cr.cvai.value:.2f}%  /  {cr.cephalic_index.value:.2f}",
        f"  patient age      : {cr.patient_age_months.value} months, "
        f"target {cr.target_duration_weeks.value} weeks",
        f"  scan landmarks   : {len(case.scan_record.landmark_set.landmarks)}  "
        f"({case.scan_record.landmark_set.frame} frame)",
        f"  measurements     : {len(case.scan_record.measurements)}",
        f"  design versions  : {len(case.design_versions)} "
        f"(latest v{latest.version if latest else 0}, "
        f"{len(latest.semantic_objects) if latest else 0} semantic objects)",
        f"  audit events     : {len(case.audit_events)}",
        "  artifacts:",
    ]
    for name, h in (latest.output_artifacts if latest else {}).items():
        lines.append(f"    {name:24s} {h}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo = CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")
        case = build_demo_case(repo)

        case_id = repo.save_case(case)
        reloaded = repo.load_case(case_id)

        assert reloaded == case, (
            "LOSSLESS ROUND-TRIP FAILED: reloaded Case is not deep-equal to the "
            "original. The data model has a serialization bug."
        )
        # Determinism: re-serialize the reloaded case identically.
        assert case.model_dump_json() == reloaded.model_dump_json()

        EXAMPLES.mkdir(exist_ok=True)
        (EXAMPLES / "demo_case.json").write_text(
            reloaded.model_dump_json(indent=2), encoding="utf-8"
        )

        sys.stdout.write(_summary(reloaded) + "\n")
        sys.stdout.write("deep-equality OK | deterministic OK | wrote examples/demo_case.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
