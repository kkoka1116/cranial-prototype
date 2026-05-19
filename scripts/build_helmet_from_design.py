"""Week-4 integration proof: hand-authored Design -> patient-specific helmet.

End to end, no LLM:
  real Week 1+2 ScanRecord -> hand-authored 5-object Design -> synthesize
  -> assemble + persist a Case -> reload (deep-equal) -> synthesize again
  (identical helmet_hash) -> export STL -> print the audit chain.

Run: uv run python scripts/build_helmet_from_design.py [--write]
(--write refreshes examples/hand_authored_design.json; default runs into a
temp dir so a normal run never dirties the working tree.)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from anatomy.cleanup import clean_scan
from anatomy.config import LANDMARK_ORDER
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import all_measurements
from anatomy.registration import apply_transform, register
from anatomy.synthetic import derive_landmarks_from_synthetic
from datamodel.case import Case
from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.design import Design
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.scan_record import ScanRecord
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    StrapMount,
    TrimLine,
    VentRegion,
)
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head
from storage.repository import CaseRepository
from synthesis.pipeline import synthesize

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

_HEAD = HeadConfig(
    occipital_flat_mm=15.0,
    occipital_flat_radius_mm=55.0,
    occipital_flat_lateral_offset_mm=35.0,
)


def _prov(source: ProvenanceSource, rationale: str, conf: float | None = 0.9) -> Provenance:
    return Provenance(source=source, rationale=rationale, confidence=conf)


def _tv(value, source: ProvenanceSource, rationale: str):
    return TracedValue(value=value, provenance=_prov(source, rationale))


def _build_scan_record(repo: CaseRepository) -> ScanRecord:
    derived = derive_landmarks_from_synthetic(_HEAD)
    source = LandmarkSet(
        frame="source",
        landmarks={
            n: Landmark(name=n, position_mm=derived.landmarks[n].position_mm, method="manual")
            for n in LANDMARK_ORDER
        },
    )
    raw = generate_test_head(_HEAD)
    cleaned = clean_scan(raw)
    transform, canonical = register(source)
    registered = apply_transform(cleaned.mesh, transform)
    mesh_hash = repo.save_blob(registered.export(file_type="stl"), "model/stl")
    return ScanRecord(
        mesh_hash=mesh_hash,
        source_filename="synthetic_moderate_right_plagio.stl",
        landmark_set=canonical,
        measurements=list(all_measurements(registered, canonical).values()),
        cleanup_report=cleaned.report,
        registered_transform=[[float(x) for x in row] for row in transform.tolist()],
    )


def _build_design() -> Design:
    clin = ProvenanceSource.CLINICAL_INPUT
    tmpl = ProvenanceSource.TEMPLATE_DEFAULT
    correction = CorrectionZone(
        anatomical_anchors=["opisthocranion", "euryon_right"],
        region_ref="opisthocranion",
        target_offset_mm=_tv(
            0.8,
            clin,
            "remodel the right-posterior flattening; bounded so the corrected wall stays >= the 3 mm minimum",
        ),
        pressure_target_mmHg=_tv(12.0, clin, "gentle, infant-safe remodeling pressure"),
        schedule=_tv("progressive", tmpl, "ramp pressure to improve tolerance"),
        duration_weeks=_tv(12, clin, "typical course for moderate plagiocephaly"),
        transition_smoothing_mm=_tv(22.0, tmpl, "smooth blend, no pressure ridge"),
        falloff_function=_tv("gaussian", tmpl, "smooth radial falloff"),
    )
    relief = ReliefZone(
        anatomical_anchors=["glabella"],
        region_ref="glabella",
        relief_offset_mm=_tv(2.0, clin, "clearance over the contralateral frontal bossing"),
        reason=_tv("do not press the prominent left forehead", clin, "intake exam finding"),
        transition_smoothing_mm=_tv(18.0, tmpl, "smooth blend out of the relief"),
    )
    trim = TrimLine(
        anatomical_anchors=["tragion_left", "tragion_right"],
        landmark_path=["glabella", "tragion_right", "opisthocranion", "tragion_left"],
        height_offsets_mm=_tv(
            [-8.0, -12.0, -10.0, -12.0], tmpl, "clear ears and brow per template"
        ),
        edge_treatment=_tv("rolled", tmpl, "rolled edge for skin comfort"),
        edge_thickness_mm=_tv(2.5, tmpl, "manufacturable edge thickness"),
    )
    vent = VentRegion(
        anatomical_anchors=["vertex"],
        region_ref="vertex",
        pattern=_tv("hex_lattice", tmpl, "standard ventilation pattern"),
        open_area_fraction=_tv(0.15, tmpl, "ventilation vs structural integrity balance"),
        feature_size_mm=_tv(6.0, tmpl, "manufacturable hole size"),
    )
    strap = StrapMount(
        anatomical_anchors=["tragion_left"],
        position_ref="tragion_left",
        strap_width_mm=_tv(20.0, tmpl, "standard chin-strap width"),
        angle_deg=_tv(35.0, tmpl, "strap exit angle toward the chin"),
        reinforcement_radius_mm=_tv(8.0, tmpl, "fillet to avoid a stress riser"),
    )
    return Design(
        version=1,
        semantic_objects=[correction, relief, trim, vent, strap],
        change_summary="hand-authored plagiocephaly design (stands in for the week-6 LLM)",
    )


def _clinical(scan: ScanRecord) -> CranialClinicalRecord:
    by = {m.name: m for m in scan.measurements}
    cvai = round(by["cvai"].value, 2) if "cvai" in by else 7.5
    ci = round(by["cephalic_index"].value, 2) if "cephalic_index" in by else 80.0
    return CranialClinicalRecord(
        patient_age_months=_tv(7, ProvenanceSource.CLINICAL_INPUT, "age at intake from referral"),
        provenance=_prov(ProvenanceSource.CLINICAL_INPUT, "record authored at intake exam"),
        diagnosis=_tv(
            CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY,
            ProvenanceSource.CLINICAL_INPUT,
            "right posterior flattening with contralateral frontal bossing on exam",
        ),
        cvai=TracedValue(
            value=cvai,
            provenance=_prov(ProvenanceSource.KERNEL_DERIVATION, f"CVAI {cvai}% from scan"),
        ),
        cephalic_index=TracedValue(
            value=ci,
            provenance=_prov(ProvenanceSource.KERNEL_DERIVATION, f"cephalic index {ci} from scan"),
        ),
        severity=_tv("moderate", ProvenanceSource.CLINICAL_INPUT, "CVAI in the moderate band"),
        laterality=_tv("right", ProvenanceSource.CLINICAL_INPUT, "flattening is right-posterior"),
        target_duration_weeks=_tv(12, ProvenanceSource.CLINICAL_INPUT, "typical moderate course"),
        asymmetry_pattern=_tv(
            "right occipital flattening, left frontal bossing",
            ProvenanceSource.CLINICAL_INPUT,
            "documented asymmetry pattern",
        ),
        contraindications=[],
    )


def _audit_chain(case: Case) -> str:
    design = case.latest_design()
    lines = ["", "AUDIT CHAIN  (geometry  <-  operation  <-  clinical decision)", "=" * 70]
    by_sem: dict[str, list] = {}
    for op in design.operations:
        for sem in op.derived_from:
            by_sem.setdefault(str(sem), []).append(op)
    for obj in design.semantic_objects:
        lines.append(f"\n[{obj.type}]  id={obj.id}")
        # Surface one representative traced rationale per object.
        for fld in (
            "target_offset_mm",
            "relief_offset_mm",
            "edge_treatment",
            "pattern",
            "strap_width_mm",
        ):
            tvv = getattr(obj, fld, None)
            if tvv is not None:
                lines.append(
                    f'    {fld} = {tvv.value!r}  <-  "{tvv.provenance.rationale}"'
                    f"  ({tvv.provenance.source.value})"
                )
                break
        for op in by_sem.get(str(obj.id), []):
            lines.append(f"    -> kernel op: {op.op_type}  (op_id={op.op_id})")
    lines.append("\nconstraints captured from the Week-1 validators:")
    for c in design.constraints_checked:
        flag = "ok " if c.satisfied else "XX "
        lines.append(f"    [{flag}] {c.name}: actual={c.actual_value} limit={c.limit_value}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_helmet_from_design")
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="refresh examples/hand_authored_design.json (default: temp, no git churn)",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        repo = CaseRepository(tp / "cases.db", tp / "blobs")

        scan = _build_scan_record(repo)
        design = _build_design()

        result = synthesize(design, scan, repo)

        versioned = design.model_copy(
            update={
                "operations": result.operations,
                "constraints_checked": result.constraints,
                "output_artifacts": {
                    "helmet_stl": result.helmet_hash,
                    "registered_scan_stl": scan.mesh_hash,
                },
            }
        )
        case = Case(
            device_class="cranial_helmet",
            clinical_narrative=(
                "7-month-old, right posterior positional plagiocephaly with mild "
                "contralateral frontal bossing. Referred for a cranial remolding helmet."
            ),
            clinical_record=_clinical(scan),
            scan_record=scan,
        )
        case.add_design_version(versioned)
        for ev in result.audit_events:
            case.append_audit(ev)

        case_id = repo.save_case(case)
        reloaded = repo.load_case(case_id)
        assert reloaded == case, (
            "LOSSLESS ROUND-TRIP FAILED: reloaded Case != original (the new "
            "operation/constraint payload broke Week-3 losslessness)."
        )

        result2 = synthesize(_build_design(), scan, repo)
        assert (
            result.helmet_hash == result2.helmet_hash
        ), f"NONDETERMINISTIC SYNTHESIS: {result.helmet_hash} != {result2.helmet_hash}"

        out_stl = Path("/tmp/synth_helmet.stl")
        out_stl.write_bytes(repo.load_blob(result.helmet_hash))

        if args.write:
            EXAMPLES.mkdir(exist_ok=True)
            (EXAMPLES / "hand_authored_design.json").write_text(
                versioned.model_dump_json(indent=2), encoding="utf-8"
            )
            dest = "examples/hand_authored_design.json"
        else:
            (tp / "hand_authored_design.json").write_text(
                versioned.model_dump_json(indent=2), encoding="utf-8"
            )
            dest = "(temp; pass --write to refresh examples/hand_authored_design.json)"

        sys.stdout.write(_audit_chain(reloaded) + "\n")
        sys.stdout.write(
            f"\ndeep-equality OK | deterministic OK ({result.helmet_hash})\n"
            f"helmet STL -> {out_stl} | design JSON -> {dest}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
