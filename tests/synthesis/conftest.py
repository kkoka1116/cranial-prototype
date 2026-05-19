"""Shared synthesis fixtures: build a (repo, ScanRecord) for a HeadConfig.

Mirrors the Week-3 demo plumbing: generate a synthetic head, derive its
canonical landmarks, store the mesh as a content-addressed blob, and wrap
the Week-2 outputs in a ScanRecord. No committed binaries (project
convention) — everything is generated per session into a tmp repo.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from anatomy.cleanup import CleanupReport
from anatomy.synthetic import derive_landmarks_from_synthetic
from datamodel.scan_record import ScanRecord
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head
from storage.repository import CaseRepository

_IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _scan_record_for(repo: CaseRepository, cfg: HeadConfig, name: str) -> ScanRecord:
    head = generate_test_head(cfg)
    mesh_hash = repo.save_blob(head.export(file_type="stl"), "model/stl")
    return ScanRecord(
        mesh_hash=mesh_hash,
        source_filename=f"synthetic_{name}.stl",
        landmark_set=derive_landmarks_from_synthetic(cfg),
        measurements=[],
        cleanup_report=CleanupReport(
            was_already_manifold=True,
            duplicate_vertices_merged=0,
            degenerate_faces_removed=0,
            holes_filled=0,
            faces_removed=0,
            faces_added=0,
            normals_flipped=False,
        ),
        registered_transform=_IDENTITY,
    )


@pytest.fixture
def make_case_env(
    tmp_path: Path,
) -> Callable[[HeadConfig, str], tuple[CaseRepository, ScanRecord]]:
    """Return a factory: (HeadConfig, name) -> (repo, ScanRecord)."""

    def _factory(cfg: HeadConfig, name: str = "head") -> tuple[CaseRepository, ScanRecord]:
        repo = CaseRepository(tmp_path / f"{name}.db", tmp_path / f"blobs_{name}")
        return repo, _scan_record_for(repo, cfg, name)

    return _factory


_PLAGIO = HeadConfig(
    occipital_flat_mm=15.0,
    occipital_flat_radius_mm=55.0,
    occipital_flat_lateral_offset_mm=35.0,
)
_BASELINE = HeadConfig()


@pytest.fixture
def plagio_cfg() -> HeadConfig:
    return _PLAGIO


@pytest.fixture
def baseline_cfg() -> HeadConfig:
    return _BASELINE


def build_hand_authored_design():
    """A clinically coherent 5-object Design (fresh uuids each call).

    Magnitudes are bounded below the 4.0 mm default wall so corrections do
    not pierce the shell. Importable by tests and the demo script.
    """
    from datamodel.design import Design
    from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
    from datamodel.semantic import (
        CorrectionZone,
        ReliefZone,
        StrapMount,
        TrimLine,
        VentRegion,
    )

    def tv(value, source, rationale, confidence=0.9):
        return TracedValue(
            value=value,
            provenance=Provenance(source=source, rationale=rationale, confidence=confidence),
        )

    clin = ProvenanceSource.CLINICAL_INPUT
    tmpl = ProvenanceSource.TEMPLATE_DEFAULT

    correction = CorrectionZone(
        anatomical_anchors=["opisthocranion", "euryon_right"],
        region_ref="opisthocranion",
        target_offset_mm=tv(
            0.8,
            clin,
            "correct the right-posterior flattening; bounded so the corrected wall stays >= the 3 mm minimum",
        ),
        pressure_target_mmHg=tv(12.0, clin, "gentle remodeling pressure for an infant"),
        schedule=tv("progressive", tmpl, "ramp pressure for tolerance"),
        duration_weeks=tv(12, clin, "typical moderate-plagiocephaly course"),
        transition_smoothing_mm=tv(22.0, tmpl, "smooth blend, no pressure ridge"),
        falloff_function=tv("gaussian", tmpl, "smooth radial falloff"),
    )
    relief = ReliefZone(
        anatomical_anchors=["glabella"],
        region_ref="glabella",
        relief_offset_mm=tv(2.0, clin, "clearance over the contralateral frontal bossing"),
        reason=tv("avoid pressing the prominent left forehead", clin, "exam finding"),
        transition_smoothing_mm=tv(18.0, tmpl, "smooth blend out of relief"),
    )
    trim = TrimLine(
        anatomical_anchors=["tragion_left", "tragion_right"],
        landmark_path=["glabella", "tragion_right", "opisthocranion", "tragion_left"],
        height_offsets_mm=tv(
            [-8.0, -12.0, -10.0, -12.0], tmpl, "trim clears the ears and brow per template"
        ),
        edge_treatment=tv("rolled", tmpl, "rolled edge for skin comfort"),
        edge_thickness_mm=tv(2.5, tmpl, "manufacturable edge thickness"),
    )
    vent = VentRegion(
        anatomical_anchors=["vertex"],
        region_ref="vertex",
        pattern=tv("hex_lattice", tmpl, "standard ventilation pattern"),
        open_area_fraction=tv(0.15, tmpl, "ventilation vs structural integrity balance"),
        feature_size_mm=tv(6.0, tmpl, "manufacturable hole size"),
    )
    strap = StrapMount(
        anatomical_anchors=["tragion_left"],
        position_ref="tragion_left",
        strap_width_mm=tv(20.0, tmpl, "standard chin-strap width"),
        angle_deg=tv(35.0, tmpl, "strap exit angle toward the chin"),
        reinforcement_radius_mm=tv(8.0, tmpl, "fillet to avoid a stress riser"),
    )
    return Design(
        version=1,
        semantic_objects=[correction, relief, trim, vent, strap],
        change_summary="hand-authored plagiocephaly design (stands in for week-6 LLM)",
    )


@pytest.fixture
def design_factory():
    return build_hand_authored_design
