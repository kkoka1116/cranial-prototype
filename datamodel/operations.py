"""Kernel-operation records.

`KernelOperation` is a *data type only* this week. The translator that turns
a `CorrectionZone` into `offset_surface` operations is week 4. This week a
`KernelOperation` is either hand-authored in tests/demo or recorded from a
direct Week-1 kernel call.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class KernelOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    op_id: UUID = Field(default_factory=uuid4)
    op_type: str
    inputs: dict = Field(default_factory=dict)
    derived_from: list[UUID] = Field(default_factory=list)
