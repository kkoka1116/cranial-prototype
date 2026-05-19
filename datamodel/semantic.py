"""Semantic design objects and their discriminated union.

The five semantic-object subtypes are the design vocabulary a clinician/LLM
reasons in (week 4 will translate them into kernel operations; not this
week). Every tunable parameter is a `TracedValue` so the audit trail records
who chose it and why.

`SemanticObjectUnion` is a Pydantic v2 discriminated union keyed on `type`.
This is the single most failure-prone piece of the schema — Pydantic v2
discriminated-union deserialization has sharp edges (the discriminator must
be a `Literal` on every member, and the union must be `Annotated` with
`Field(discriminator=...)`). It is implemented and round-trip-tested first,
before anything else depends on it.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from datamodel.provenance import TracedValue


class SemanticObject(BaseModel):
    """Base for all semantic design objects.

    Concrete subtypes override `type` with a `Literal` discriminator and add
    their own fully `TracedValue`-annotated parameters.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    type: str
    anatomical_anchors: list[str] = Field(default_factory=list)


class CorrectionZone(SemanticObject):
    type: Literal["correction_zone"] = "correction_zone"
    region_ref: str
    target_offset_mm: TracedValue[float]
    # mmHg is the standard clinical pressure unit; the units-in-names
    # invariant (CLAUDE.md) requires the suffix and the brief fixes this
    # exact field name, so the mixed case is intentional.
    pressure_target_mmHg: TracedValue[float]  # noqa: N815
    schedule: TracedValue[Literal["constant", "progressive", "graduated"]]
    duration_weeks: TracedValue[int]
    transition_smoothing_mm: TracedValue[float]
    falloff_function: TracedValue[Literal["gaussian", "linear", "cosine"]]


class ReliefZone(SemanticObject):
    type: Literal["relief_zone"] = "relief_zone"
    region_ref: str
    relief_offset_mm: TracedValue[float]
    reason: TracedValue[str]
    transition_smoothing_mm: TracedValue[float]


class TrimLine(SemanticObject):
    type: Literal["trim_line"] = "trim_line"
    landmark_path: list[str]
    height_offsets_mm: TracedValue[list[float]]
    edge_treatment: TracedValue[Literal["rolled", "flared", "straight"]]
    edge_thickness_mm: TracedValue[float]


class VentRegion(SemanticObject):
    type: Literal["vent_region"] = "vent_region"
    region_ref: str
    pattern: TracedValue[Literal["hex_lattice", "circular", "slot"]]
    open_area_fraction: TracedValue[float]
    feature_size_mm: TracedValue[float]


class StrapMount(SemanticObject):
    type: Literal["strap_mount"] = "strap_mount"
    position_ref: str
    strap_width_mm: TracedValue[float]
    angle_deg: TracedValue[float]
    reinforcement_radius_mm: TracedValue[float]


# Discriminated union over `type`. A serialized list of mixed semantic
# objects round-trips back to the correct concrete subclasses.
SemanticObjectUnion = Annotated[
    CorrectionZone | ReliefZone | TrimLine | VentRegion | StrapMount,
    Field(discriminator="type"),
]
