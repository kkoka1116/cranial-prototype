"""Design — one immutable version of a helmet design."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from datamodel.constraints import Constraint
from datamodel.operations import KernelOperation
from datamodel.semantic import SemanticObjectUnion


class Design(BaseModel):
    """An immutable design version. New versions are appended to a Case via
    `Case.add_design_version`, never mutated in place."""

    model_config = ConfigDict(frozen=True)

    version: int
    semantic_objects: list[SemanticObjectUnion] = Field(default_factory=list)
    operations: list[KernelOperation] = Field(default_factory=list)
    output_artifacts: dict[str, str] = Field(default_factory=dict)
    constraints_checked: list[Constraint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_version: int | None = None
    change_summary: str | None = None
