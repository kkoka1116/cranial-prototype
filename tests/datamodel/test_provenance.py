"""Round-trip + determinism for Provenance and the generic TracedValue[T].

Provenance round-trips with prior_value as float / str / dict. TracedValue
round-trips over float, int, str, bool, list[float], and a Literal. Plus the
validator contract: empty/whitespace rationale and out-of-range confidence are
rejected; the unit-interval endpoints 0.0 and 1.0 are accepted.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from datamodel.provenance import Provenance, ProvenanceSource, TracedValue


def _prov(prior_value: object = None) -> Provenance:
    return Provenance(
        source=ProvenanceSource.CLINICIAN_OVERRIDE,
        rationale="clinician adjusted the value after reviewing the scan",
        agent_id="agent-7",
        prior_value=prior_value,
        confidence=0.75,
    )


def _assert_round_trip(obj) -> None:
    cls = type(obj)
    restored = cls.model_validate_json(obj.model_dump_json())
    assert restored == obj
    assert obj.model_dump_json() == obj.model_dump_json()
    once = obj.model_dump_json()
    assert once == cls.model_validate_json(once).model_dump_json()


@pytest.mark.parametrize(
    "prior_value",
    [
        3.14,
        "previous textual value",
        {"old": 1, "nested": {"k": [1, 2, 3]}},
    ],
)
def test_provenance_round_trips_with_each_prior_value_kind(prior_value: object) -> None:
    _assert_round_trip(_prov(prior_value))


class _TVHolder(BaseModel):
    """Container exercising several concrete TracedValue[T] parametrizations."""

    model_config = ConfigDict(frozen=True)

    f: TracedValue[float]
    i: TracedValue[int]
    s: TracedValue[str]
    b: TracedValue[bool]
    lst: TracedValue[list[float]]
    lit: TracedValue[Literal["a", "b", "c"]]


def test_traced_value_round_trips_over_many_types() -> None:
    p = _prov(prior_value="seed")
    holder = _TVHolder(
        f=TracedValue(value=2.5, provenance=p),
        i=TracedValue(value=42, provenance=p),
        s=TracedValue(value="hello", provenance=p),
        b=TracedValue(value=True, provenance=p),
        lst=TracedValue(value=[1.0, -2.0, 3.5], provenance=p),
        lit=TracedValue(value="b", provenance=p),
    )
    _assert_round_trip(holder)

    restored = _TVHolder.model_validate_json(holder.model_dump_json())
    assert restored.f.value == 2.5
    assert restored.i.value == 42
    assert restored.s.value == "hello"
    assert restored.b.value is True
    assert restored.lst.value == [1.0, -2.0, 3.5]
    assert restored.lit.value == "b"
    assert restored.f.provenance.source is ProvenanceSource.CLINICIAN_OVERRIDE


def test_traced_value_typeadapter_round_trip() -> None:
    adapter = TypeAdapter(TracedValue[bool])
    obj = TracedValue(value=False, provenance=_prov())
    dumped = adapter.dump_json(obj)
    restored = adapter.validate_json(dumped)
    assert restored.value is False
    assert adapter.dump_json(restored) == dumped


def test_empty_or_whitespace_rationale_rejected() -> None:
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError):
            Provenance(source=ProvenanceSource.LLM_PROPOSAL, rationale=bad)


def test_confidence_out_of_range_rejected() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Provenance(
                source=ProvenanceSource.LLM_PROPOSAL,
                rationale="proposed by the model",
                confidence=bad,
            )


def test_confidence_unit_interval_endpoints_accepted() -> None:
    for good in (0.0, 1.0):
        p = Provenance(
            source=ProvenanceSource.LLM_PROPOSAL,
            rationale="proposed by the model",
            confidence=good,
        )
        assert p.confidence == good
        _assert_round_trip(p)
