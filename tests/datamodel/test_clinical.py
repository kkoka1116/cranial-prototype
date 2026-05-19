"""Round-trip + determinism for CranialClinicalRecord.

Two fully-populated instances: one with a real laterality + str
asymmetry_pattern, one with laterality=None + asymmetry_pattern=None. Asserts
provenance survives the round-trip, and re-asserts the Provenance validator
contract (empty rationale / out-of-range confidence) at this layer.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datamodel.clinical import CranialClinicalRecord, CranialDiagnosis
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue


def _prov(src: ProvenanceSource = ProvenanceSource.CLINICAL_INPUT) -> Provenance:
    return Provenance(
        source=src,
        rationale="recorded from the clinical intake form",
        agent_id=None,
        confidence=0.9,
    )


def _tv(value):
    return TracedValue(value=value, provenance=_prov())


def _record(*, laterality, asymmetry_pattern) -> CranialClinicalRecord:
    return CranialClinicalRecord(
        patient_age_months=_tv(7),
        notes="referred by pediatrician; first helmet",
        provenance=_prov(ProvenanceSource.CLINICIAN_OVERRIDE),
        diagnosis=_tv(CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY),
        cvai=_tv(7.5),
        cephalic_index=_tv(80.4),
        severity=_tv("moderate"),
        laterality=_tv(laterality),
        target_duration_weeks=_tv(12),
        asymmetry_pattern=_tv(asymmetry_pattern),
        contraindications=["scalp lesion", "uncontrolled reflux"],
    )


def _assert_round_trip(obj: CranialClinicalRecord) -> None:
    restored = CranialClinicalRecord.model_validate_json(obj.model_dump_json())
    assert restored == obj
    assert obj.model_dump_json() == obj.model_dump_json()
    once = obj.model_dump_json()
    assert once == CranialClinicalRecord.model_validate_json(once).model_dump_json()


def test_cranial_record_round_trips_with_real_laterality() -> None:
    rec = _record(laterality="right", asymmetry_pattern="right posterior flattening")
    _assert_round_trip(rec)
    restored = CranialClinicalRecord.model_validate_json(rec.model_dump_json())
    assert restored.laterality.value == "right"
    assert restored.asymmetry_pattern.value == "right posterior flattening"
    assert restored.cvai.provenance.source is ProvenanceSource.CLINICAL_INPUT
    assert restored.diagnosis.value is CranialDiagnosis.POSITIONAL_PLAGIOCEPHALY
    assert restored.device_class == "cranial_helmet"


def test_cranial_record_round_trips_with_none_optionals() -> None:
    rec = _record(laterality=None, asymmetry_pattern=None)
    _assert_round_trip(rec)
    restored = CranialClinicalRecord.model_validate_json(rec.model_dump_json())
    assert restored.laterality.value is None
    assert restored.asymmetry_pattern.value is None
    assert restored.laterality.provenance.rationale == rec.laterality.provenance.rationale


def test_provenance_validator_contract_holds_at_clinical_layer() -> None:
    with pytest.raises(ValidationError):
        Provenance(source=ProvenanceSource.CLINICAL_INPUT, rationale="   ")
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Provenance(
                source=ProvenanceSource.CLINICAL_INPUT,
                rationale="intake",
                confidence=bad,
            )
