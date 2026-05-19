"""AuditEvent validation + round-trip.

Covers the `action` non-empty validator (parity with the well-tested
Provenance.rationale validator) and lossless/deterministic serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datamodel.audit import AuditActor, AuditEvent


def test_action_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(actor=AuditActor.SYSTEM, action="")


def test_action_must_be_non_whitespace() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(actor=AuditActor.SYSTEM, action="   ")


def test_audit_event_round_trips_lossless_and_deterministic() -> None:
    ev = AuditEvent(
        actor=AuditActor.LLM_AGENT,
        action="proposed_parameters",
        payload={"zone": "rpq", "offset_mm": 6.0},
        rationale="hand-authored fixture",
    )
    restored = AuditEvent.model_validate_json(ev.model_dump_json())
    assert restored == ev
    once = ev.model_dump_json()
    assert once == ev.model_dump_json()
    assert once == AuditEvent.model_validate_json(once).model_dump_json()
