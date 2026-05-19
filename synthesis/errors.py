"""Synthesis exception hierarchy.

Per invariant #12, anchor resolution and execution fail loudly with
specific, informative errors — never silent degradation.
"""

from __future__ import annotations


class SynthesisError(Exception):
    """Base class for all synthesis-package errors."""


class AnchorResolutionError(SynthesisError):
    """A semantic object's anchors could not be resolved against the
    patient's LandmarkSet (missing landmark, or geometrically insufficient
    anchors). Carries the offending object id and what was insufficient.
    """


class ExecutionError(SynthesisError):
    """A phase produced non-manifold/empty geometry or a kernel call failed.
    Carries the offending operation and the semantic object id it derived
    from."""
