"""Provenance and the generic TracedValue[T] wrapper.

`TracedValue[T]` wraps a value with the `Provenance` recording who decided
it and why. It is applied to clinical-record fields and semantic-object
parameters — values where "who decided this and why" is regulatorily
critical. It is deliberately NOT retrofitted onto `anatomy.Landmark` (which
already carries its own `method`/`confidence`); see CLAUDE.md reconciliation
principle.

Determinism (invariant #10): a constructed instance has a fixed `timestamp`;
serializing the *same* instance twice is byte-identical. Non-determinism
only arises across distinct constructions (different wall-clock), which is
expected and orthogonal to the round-trip guarantee.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T")


class ProvenanceSource(str, Enum):
    """Where a traced value originated."""

    CLINICAL_INPUT = "clinical_input"
    LANDMARK_DETECTION = "landmark_detection"
    LLM_PROPOSAL = "llm_proposal"
    CLINICIAN_OVERRIDE = "clinician_override"
    TEMPLATE_DEFAULT = "template_default"
    VALIDATOR_CORRECTION = "validator_correction"
    KERNEL_DERIVATION = "kernel_derivation"


class Provenance(BaseModel):
    """Who decided a value, and why.

    `rationale` is required and must be non-empty — a provenance with no
    rationale is meaningless for the audit trail. `confidence`, when present,
    is a probability in [0, 1].
    """

    model_config = ConfigDict(frozen=True)

    source: ProvenanceSource
    rationale: str
    agent_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prior_value: float | str | dict | None = None
    confidence: float | None = None

    @field_validator("rationale")
    @classmethod
    def _rationale_non_empty(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("Provenance.rationale must be a non-empty string")
        return v

    @field_validator("confidence")
    @classmethod
    def _confidence_in_unit_interval(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Provenance.confidence must be in [0, 1], got {v}")
        return v


class TracedValue(BaseModel, Generic[T]):
    """A value plus the provenance of how it was decided.

    Generic over the wrapped type so clinical/semantic models can declare
    `TracedValue[float]`, `TracedValue[CranialDiagnosis]`, etc., and the
    discriminator/round-trip machinery resolves the concrete type.
    """

    model_config = ConfigDict(frozen=True)

    value: T
    provenance: Provenance

    @model_validator(mode="before")
    @classmethod
    def _coerce_any_parametrization(cls, data: object) -> object:
        """Normalize any TracedValue-like input to a plain dict so the
        field's *concrete* `T` performs the value validation.

        Without this, Pydantic v2 requires the input's generic
        parametrization to match the field exactly — e.g. a
        ``TracedValue[str]`` instance is rejected for a
        ``TracedValue[Literal[...]]`` field, and a bare
        ``TracedValue(value=..., provenance=...)`` is rejected for any
        ``TracedValue[X]`` field. Coercing to ``{"value", "provenance"}``
        lets the destination type drive validation, which keeps
        construction ergonomic everywhere (tests, demo script, clinical
        and semantic models) while preserving round-trip correctness.
        """
        if isinstance(data, TracedValue):
            return {"value": data.value, "provenance": data.provenance}
        return data
