"""Storage determinism: save->load->save->load is byte-identical, and
blob hashes are stable across round trips (invariant #10)."""

from __future__ import annotations

from pathlib import Path

from datamodel.case import Case
from storage.repository import CaseRepository


def test_double_round_trip_is_byte_identical(tmp_path: Path, populated_case: Case) -> None:
    repo = CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")
    cid = repo.save_case(populated_case)

    loaded_1 = repo.load_case(cid)
    repo.save_case(loaded_1)
    loaded_2 = repo.load_case(cid)

    assert loaded_1 == loaded_2
    assert loaded_1.model_dump_json() == loaded_2.model_dump_json()
    # And equal to the original in-memory case.
    assert loaded_1.model_dump_json() == populated_case.model_dump_json()


def test_blob_hashes_stable_across_round_trip(tmp_path: Path) -> None:
    repo = CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")
    content = b"a registered mesh, deterministically exported"
    h_first = repo.save_blob(content, "model/stl")

    # New repository instance over the same dirs (simulates reload).
    repo2 = CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")
    h_again = repo2.save_blob(content, "model/stl")
    assert h_first == h_again
    assert repo2.load_blob(h_again) == content


def test_case_json_column_matches_model_dump(tmp_path: Path, populated_case: Case) -> None:
    """The persisted JSON is exactly Case.model_dump_json() — no lossy
    re-encoding in the storage layer."""
    repo = CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")
    cid = repo.save_case(populated_case)
    reloaded = repo.load_case(cid)
    assert reloaded.model_dump_json() == populated_case.model_dump_json()
