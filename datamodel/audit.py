"""Audit events — the append-only trail of who did what to a Case."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AuditActor(str, Enum):
    CLINICIAN = "clinician"
    SYSTEM = "system"
    LLM_AGENT = "llm_agent"
    KERNEL = "kernel"
    VALIDATOR = "validator"


class AuditEvent(BaseModel):
    """One entry in a Case's audit trail. Immutable once created."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: AuditActor
    action: str
    target_id: UUID | None = None
    payload: dict = Field(default_factory=dict)
    rationale: str | None = None

    @field_validator("action")
    @classmethod
    def _action_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("AuditEvent.action must be a non-empty string")
        return v
