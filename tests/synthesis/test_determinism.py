"""synthesize is deterministic: identical inputs -> identical helmet_hash,
in-process and across separate processes (acceptance #5)."""

from __future__ import annotations

import subprocess
import sys

from synthesis.pipeline import synthesize


def test_synthesize_identical_helmet_hash_in_process(make_case_env, plagio_cfg, design_factory):
    repo, scan = make_case_env(plagio_cfg, "plagio")
    design = design_factory()
    r1 = synthesize(design, scan, repo)
    r2 = synthesize(design, scan, repo)
    assert r1.helmet_hash == r2.helmet_hash
    # Operation *content* identical (op_id is uuid4-defaulted; excluded).
    assert len(r1.operations) == len(r2.operations)
    for a, b in zip(r1.operations, r2.operations, strict=True):
        assert (a.op_type, a.inputs, a.derived_from) == (b.op_type, b.inputs, b.derived_from)


_CROSS_PROCESS_SNIPPET = """
import io, sys, tempfile
from pathlib import Path
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head
from anatomy.synthetic import derive_landmarks_from_synthetic
from anatomy.cleanup import CleanupReport
from datamodel.scan_record import ScanRecord
from storage.repository import CaseRepository
from synthesis.pipeline import synthesize
sys.path.insert(0, "tests/synthesis")
from conftest import build_hand_authored_design  # noqa: E402

cfg = HeadConfig(occipital_flat_mm=15.0, occipital_flat_radius_mm=55.0,
                 occipital_flat_lateral_offset_mm=35.0)
with tempfile.TemporaryDirectory() as t:
    tp = Path(t)
    repo = CaseRepository(tp / "c.db", tp / "b")
    head = generate_test_head(cfg)
    mh = repo.save_blob(head.export(file_type="stl"), "model/stl")
    sr = ScanRecord(
        mesh_hash=mh, source_filename="s.stl",
        landmark_set=derive_landmarks_from_synthetic(cfg), measurements=[],
        cleanup_report=CleanupReport(was_already_manifold=True,
            duplicate_vertices_merged=0, degenerate_faces_removed=0,
            holes_filled=0, faces_removed=0, faces_added=0, normals_flipped=False),
        registered_transform=[[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]])
    print(synthesize(build_hand_authored_design(), sr, repo).helmet_hash)
"""


def test_synthesize_deterministic_across_processes() -> None:
    def run() -> str:
        out = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_SNIPPET],
            capture_output=True,
            text=True,
            check=True,
            cwd=".",
        )
        return out.stdout.strip().splitlines()[-1]

    assert run() == run()
