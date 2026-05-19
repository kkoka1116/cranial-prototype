"""Storage exception hierarchy."""

from __future__ import annotations


class StorageError(Exception):
    """Base class for all storage-package errors."""


class CaseNotFoundError(StorageError):
    """No case with the requested id exists in the repository."""


class BlobNotFoundError(StorageError):
    """No blob with the requested hash exists in the blob store."""
