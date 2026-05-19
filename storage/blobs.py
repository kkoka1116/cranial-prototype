"""Content-addressed blob store.

Bytes are keyed by the SHA-256 of their content, so storing identical
content twice is a no-op that returns the same hash (the dedup /
determinism guarantee for invariant #9 and #10). Hashes are returned
prefixed `sha256:`; on disk the filename is the bare hex digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from storage.errors import BlobNotFoundError

_PREFIX = "sha256:"


def _strip(hash_str: str) -> str:
    return hash_str[len(_PREFIX) :] if hash_str.startswith(_PREFIX) else hash_str


class BlobStore:
    def __init__(self, blob_dir: Path) -> None:
        self.blob_dir = Path(blob_dir)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, hash_str: str) -> Path:
        return self.blob_dir / _strip(hash_str)

    def put(self, content: bytes, media_type: str) -> str:
        """Store `content`, return its `sha256:` hash. Idempotent: identical
        content is written at most once (no duplicate file)."""
        digest = hashlib.sha256(content).hexdigest()
        hash_str = _PREFIX + digest
        path = self._path(hash_str)
        if not path.exists():
            # Write atomically so a crash mid-write can't leave a partial
            # blob under a name that claims to be its hash.
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        return hash_str

    def get(self, hash_str: str) -> bytes:
        path = self._path(hash_str)
        if not path.exists():
            raise BlobNotFoundError(f"no blob for hash {hash_str!r}")
        return path.read_bytes()

    def exists(self, hash_str: str) -> bool:
        return self._path(hash_str).exists()
