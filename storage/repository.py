"""CaseRepository — SQLite (one JSON column + metadata) plus the blob store.

A prototype that stores serialized Pydantic JSON in a SQLite column with
metadata columns for indexing does not need an ORM, and an ORM would muddy
the determinism story (CLAUDE.md). Correctness over speed:
`append_audit_event` deliberately load-modify-saves the whole case.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from datamodel.audit import AuditEvent
from datamodel.case import Case
from storage.blobs import BlobStore
from storage.errors import CaseNotFoundError

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class CaseSummary(BaseModel):
    """Lightweight case list-view row (no full Case deserialization)."""

    case_id: str
    device_class: str
    created_at: str
    updated_at: str
    design_version_count: int


class CaseRepository:
    def __init__(self, db_path: Path, blob_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(blob_dir)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(ddl)

    # -- cases ------------------------------------------------------------- #

    def save_case(self, case: Case) -> str:
        """Upsert a case keyed by `case_id` (saving twice updates, never
        duplicates). Returns the case_id."""
        case_id = str(case.case_id)
        payload = case.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, device_class, created_at, updated_at, case_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    device_class = excluded.device_class,
                    created_at   = excluded.created_at,
                    updated_at   = excluded.updated_at,
                    case_json    = excluded.case_json
                """,
                (
                    case_id,
                    case.device_class,
                    case.created_at.isoformat(),
                    case.updated_at.isoformat(),
                    payload,
                ),
            )
        return case_id

    def load_case(self, case_id: str) -> Case:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT case_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise CaseNotFoundError(f"no case with id {case_id!r}")
        return Case.model_validate_json(row[0])

    def list_cases(self) -> list[CaseSummary]:
        """Summaries without deserializing full Case objects. The design
        version count is read via SQLite JSON1, not Pydantic."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT case_id, device_class, created_at, updated_at,
                       json_array_length(json_extract(case_json, '$.design_versions'))
                FROM cases
                ORDER BY created_at
                """
            ).fetchall()
        return [
            CaseSummary(
                case_id=r[0],
                device_class=r[1],
                created_at=r[2],
                updated_at=r[3],
                design_version_count=r[4] or 0,
            )
            for r in rows
        ]

    # -- blobs ------------------------------------------------------------- #

    def save_blob(self, content: bytes, media_type: str) -> str:
        hash_str = self.blobs.put(content, media_type)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO blobs (hash, media_type, size_bytes, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (hash_str, media_type, len(content), datetime.utcnow().isoformat()),
            )
        return hash_str

    def load_blob(self, hash_str: str) -> bytes:
        return self.blobs.get(hash_str)

    # -- audit ------------------------------------------------------------- #

    def append_audit_event(self, case_id: str, event: AuditEvent) -> None:
        """Load-modify-save the whole case (correct, not optimized)."""
        case = self.load_case(case_id)
        case.append_audit(event)
        self.save_case(case)
