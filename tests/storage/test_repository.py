"""CaseRepository: save/load deep-equality, upsert, list, not-found."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamodel.case import Case
from storage.errors import CaseNotFoundError
from storage.repository import CaseRepository


def _repo(tmp_path: Path) -> CaseRepository:
    return CaseRepository(tmp_path / "cases.db", tmp_path / "blobs")


def test_save_load_deep_equality(tmp_path: Path, populated_case: Case) -> None:
    repo = _repo(tmp_path)
    cid = repo.save_case(populated_case)
    loaded = repo.load_case(cid)
    assert loaded == populated_case


def test_upsert_updates_not_duplicates(tmp_path: Path, populated_case: Case) -> None:
    repo = _repo(tmp_path)
    cid = repo.save_case(populated_case)
    # Mutate and re-save under the same id.
    populated_case.clinical_narrative = "UPDATED narrative"
    repo.save_case(populated_case)
    assert len(repo.list_cases()) == 1
    assert repo.load_case(cid).clinical_narrative == "UPDATED narrative"


def test_list_cases_summaries_without_full_deserialization(
    tmp_path: Path, populated_case: Case
) -> None:
    repo = _repo(tmp_path)
    repo.save_case(populated_case)
    summaries = repo.list_cases()
    assert len(summaries) == 1
    s = summaries[0]
    assert s.case_id == str(populated_case.case_id)
    assert s.device_class == "cranial_helmet"
    assert s.design_version_count == len(populated_case.design_versions) == 1


def test_load_missing_raises(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(CaseNotFoundError):
        repo.load_case("00000000-0000-0000-0000-000000000000")


def test_append_audit_event_persists(tmp_path: Path, populated_case: Case) -> None:
    from datamodel.audit import AuditActor, AuditEvent

    repo = _repo(tmp_path)
    cid = repo.save_case(populated_case)
    before = len(repo.load_case(cid).audit_events)
    repo.append_audit_event(
        cid, AuditEvent(actor=AuditActor.SYSTEM, action="exported", rationale="pdf")
    )
    after = repo.load_case(cid)
    assert len(after.audit_events) == before + 1
    assert after.audit_events[-1].action == "exported"


def test_save_and_load_blob(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    content = b"mesh stl bytes"
    h = repo.save_blob(content, "model/stl")
    assert repo.load_blob(h) == content
    # Index row exists and is idempotent.
    h2 = repo.save_blob(content, "model/stl")
    assert h == h2
