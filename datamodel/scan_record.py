"""ScanRecord — composes the real Week-2 anatomy outputs.

Reconciliation with the brief's spec (reported before implementation):
Week 2 never built an `anatomy.ReferenceFrame` type. It represents the
canonical frame two ways: a string tag on `LandmarkSet.frame`
("source"|"canonical"), and the 4×4 source→canonical matrix returned by
`anatomy.registration.register()` (a numpy array). Per the brief's explicit
fallback ("…or a list[list[float]] 4×4 matrix if Week 2 represented it that
way — match whatever Week 2 actually built"), there is no separate
`reference_frame` field: `registered_transform` carries the frame relative
to source, and the frame *state* lives in the composed `LandmarkSet.frame`.
The numpy→nested-list conversion is a datamodel-side adapter (where the
brief says adapters belong); Week-2 types are composed, never redefined.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from anatomy.cleanup import CleanupReport
from anatomy.landmarks import LandmarkSet
from anatomy.measurements import MeasurementResult


class ScanRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    mesh_hash: str
    source_filename: str
    landmark_set: LandmarkSet
    measurements: list[MeasurementResult]
    cleanup_report: CleanupReport
    registered_transform: list[list[float]]
