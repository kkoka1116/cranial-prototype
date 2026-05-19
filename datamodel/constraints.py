"""Constraint records.

A `Constraint` is a *data type only* this week — a recorded check result.
Filling constraints from a multi-domain solver is week 7; recording a Week-1
kernel validator's output as a `Constraint` is recording, not propagation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Constraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    domain: Literal["clinical", "biomechanical", "manufacturing", "regulatory"]
    expression: str
    satisfied: bool
    actual_value: float | None = None
    limit_value: float | None = None
    severity: Literal["error", "warning", "info"] = "error"
