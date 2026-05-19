"""Clinical record models.

`ClinicalRecord` is the device-agnostic base; `CranialClinicalRecord` is the
cranial-helmet specialization. Regulatorily-critical fields (diagnosis,
CVAI, severity, …) are `TracedValue`s so the audit trail records who decided
each and why.

Note on losslessness (invariant #10): `Case.clinical_record` is typed as the
concrete `CranialClinicalRecord`, NOT the base `ClinicalRecord`. A
base-typed field would deserialize back to the base and silently drop the
cranial-specific fields, breaking the lossless round-trip guarantee. When
additional device classes appear, this becomes a `device_class`-
discriminated union (mirroring `SemanticObjectUnion`).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from datamodel.provenance import Provenance, TracedValue


class ClinicalRecord(BaseModel):
    """Device-agnostic clinical record base."""

    model_config = ConfigDict(frozen=True)

    device_class: str
    patient_age_months: TracedValue[int]
    notes: str | None = None
    provenance: Provenance


class CranialDiagnosis(str, Enum):
    POSITIONAL_PLAGIOCEPHALY = "positional_plagiocephaly"
    BRACHYCEPHALY = "brachycephaly"
    SCAPHOCEPHALY = "scaphocephaly"
    POST_SYNOSTOSIS_REPAIR = "post_synostosis_repair"


class CranialClinicalRecord(ClinicalRecord):
    """Cranial-helmet clinical record."""

    device_class: Literal["cranial_helmet"] = "cranial_helmet"
    diagnosis: TracedValue[CranialDiagnosis]
    cvai: TracedValue[float]
    cephalic_index: TracedValue[float]
    severity: TracedValue[Literal["mild", "moderate", "severe"]]
    laterality: TracedValue[Literal["left", "right", "bilateral"] | None]
    target_duration_weeks: TracedValue[int]
    asymmetry_pattern: TracedValue[str | None]
    contraindications: list[str] = Field(default_factory=list)
