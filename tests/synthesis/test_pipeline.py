"""Full pipeline: Design + ScanRecord -> valid helmet; lossless Case round-trip."""

from __future__ import annotations

import io

import trimesh

from datamodel.case import Case
from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from synthesis.pipeline import synthesize


def _clinical() -> CranialClinicalRecord:
    def tv(v):
        return TracedValue(
            value=v,
            provenance=Provenance(
                source=ProvenanceSource.CLINICAL_INPUT, rationale="fixture", confidence=0.9
            ),
        )

    return CranialClinicalRecord(
        patient_age_months=tv(7),
        provenance=Provenance(
            source=ProvenanceSource.CLINICAL_INPUT, rationale="record authored at intake"
        ),
        diagnosis=tv(CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY),
        cvai=tv(7.5),
        cephalic_index=tv(80.4),
        severity=tv("moderate"),
        laterality=tv("right"),
        target_duration_weeks=tv(12),
        asymmetry_pattern=tv("right occipital flattening"),
        contraindications=[],
    )


def test_full_pipeline_valid_helmet_and_lossless_case(make_case_env, plagio_cfg, design_factory):
    repo, scan = make_case_env(plagio_cfg, "plagio")
    design = design_factory()

    result = synthesize(design, scan, repo)

    # Helmet retrievable + valid. STL is a triangle soup; weld before
    # checking watertightness (the executor already validated the mesh it
    # produced — this just reconstructs shared vertices after STL I/O).
    helmet = trimesh.load(
        file_obj=io.BytesIO(repo.load_blob(result.helmet_hash)),
        file_type="stl",
        process=True,
    )
    helmet.merge_vertices()
    assert helmet.is_watertight and helmet.is_volume

    # Invariant #11: every emitted operation traces to a semantic object.
    assert result.operations
    for op in result.operations:
        assert op.derived_from, f"{op.op_type} has empty derived_from"

    # Assemble + persist a Case with the synthesized design version.
    design = design.model_copy(
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
        clinical_narrative="7mo right posterior plagiocephaly; helmet therapy.",
        clinical_record=_clinical(),
        scan_record=scan,
    )
    case.add_design_version(design)
    for ev in result.audit_events:
        case.append_audit(ev)

    case_id = repo.save_case(case)
    reloaded = repo.load_case(case_id)
    assert reloaded == case  # Week-3 losslessness survives the new payload
    assert reloaded.latest_design().operations == result.operations


def test_pipeline_audit_chain_covers_every_object(make_case_env, plagio_cfg, design_factory):
    repo, scan = make_case_env(plagio_cfg, "plagio")
    design = design_factory()
    result = synthesize(design, scan, repo)
    # One resolve_translate audit event per semantic object, each naming it.
    rt = [e for e in result.audit_events if e.action == "resolve_translate"]
    assert len(rt) == len(design.semantic_objects)
    targeted = {str(e.target_id) for e in rt}
    assert targeted == {str(o.id) for o in design.semantic_objects}


def test_missing_landmark_fails_loudly(make_case_env, baseline_cfg, design_factory):
    """Invariant #12: a Design referencing an absent landmark raises, never
    degrades."""
    import pytest

    from datamodel.semantic import CorrectionZone
    from synthesis.errors import AnchorResolutionError

    repo, scan = make_case_env(baseline_cfg, "baseline")
    design = design_factory()
    bad = CorrectionZone(
        anatomical_anchors=["definitely_not_a_landmark"],
        region_ref="definitely_not_a_landmark",
        target_offset_mm=design.semantic_objects[0].target_offset_mm,
        pressure_target_mmHg=design.semantic_objects[0].pressure_target_mmHg,
        schedule=design.semantic_objects[0].schedule,
        duration_weeks=design.semantic_objects[0].duration_weeks,
        transition_smoothing_mm=design.semantic_objects[0].transition_smoothing_mm,
        falloff_function=design.semantic_objects[0].falloff_function,
    )
    design = design.model_copy(update={"semantic_objects": [bad]})
    with pytest.raises(AnchorResolutionError, match="definitely_not_a_landmark"):
        synthesize(design, scan, repo)
