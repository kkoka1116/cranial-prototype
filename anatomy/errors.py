"""Anatomy exception hierarchy.

AnatomyError is the base class for everything this package raises so callers
can catch a single type. Mirrors the kernel's error-hierarchy convention.
"""

from __future__ import annotations


class AnatomyError(Exception):
    """Base class for all anatomy-package errors."""


class RegistrationError(AnatomyError):
    """Landmark-based registration failed or produced an invalid frame.

    The classic cause is a left/right landmark swap, which the canonical-frame
    sign checks (glabella +Y, vertex +Z, tragion_right +X) catch.
    """


class MeasurementError(AnatomyError):
    """A measurement could not be computed (e.g., ambiguous ray intersection,
    degenerate slice) on an otherwise valid registered mesh."""
