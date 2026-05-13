"""Kernel exception hierarchy.

KernelError is the base class for all errors that escape the kernel. Anything
the kernel raises should be a subclass of this so callers can catch one type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel.validation.topology import ValidationResult


class KernelError(Exception):
    """Base class for all kernel errors."""


class ValidationError(KernelError):
    """A mesh failed a validation check.

    Carries the structured ValidationResult so callers can inspect which
    invariant was violated and by how much.
    """

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__(result.message)


class ConfigError(KernelError):
    """A user-supplied config is malformed or contradicts an invariant."""


class GeometryError(KernelError):
    """An internal geometric operation produced an unexpected result.

    Distinct from ValidationError because this represents an *internal*
    failure (e.g. a boolean op returned an empty mesh) rather than a user
    config that produced an invalid-but-well-formed result.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.context = context or {}
        super().__init__(message)
