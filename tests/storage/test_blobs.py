"""Content-addressed blob store: dedup, hashing, missing-blob error."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from storage.blobs import BlobStore
from storage.errors import BlobNotFoundError


def test_hash_is_correct_sha256(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    content = b"hello cranial world"
    h = store.put(content, "application/octet-stream")
    assert h == "sha256:" + hashlib.sha256(content).hexdigest()


def test_put_is_idempotent_and_dedups(tmp_path: Path) -> None:
    blob_dir = tmp_path / "blobs"
    store = BlobStore(blob_dir)
    content = b"deterministic mesh bytes"
    h1 = store.put(content, "model/stl")
    h2 = store.put(content, "model/stl")
    assert h1 == h2
    # Exactly one file on disk (no duplicate), ignoring any .tmp.
    files = [p for p in blob_dir.iterdir() if not p.name.endswith(".tmp")]
    assert len(files) == 1


def test_get_round_trips_content(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    content = b"\x00\x01\x02 binary \xff"
    h = store.put(content, "application/octet-stream")
    assert store.get(h) == content
    assert store.exists(h)


def test_get_missing_raises(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    missing = "sha256:" + "00" * 32
    assert not store.exists(missing)
    with pytest.raises(BlobNotFoundError):
        store.get(missing)


def test_distinct_content_distinct_hash(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    assert store.put(b"a", "t") != store.put(b"b", "t")
