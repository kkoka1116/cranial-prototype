-- Case database schema. No ORM, no migrations framework — by design.
-- Schema changes during the prototype = delete the dev DB + blob dir and
-- rebuild from scripts/build_demo_case.py.

CREATE TABLE IF NOT EXISTS cases (
    case_id      TEXT PRIMARY KEY,
    device_class TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    case_json    TEXT NOT NULL
);

-- Queryable index of blobs. The bytes themselves live on disk via the
-- content-addressed BlobStore; this table is metadata only.
CREATE TABLE IF NOT EXISTS blobs (
    hash       TEXT PRIMARY KEY,
    media_type TEXT,
    size_bytes INTEGER,
    created_at TEXT
);
