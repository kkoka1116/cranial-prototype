"""Case — the top-level aggregate and single source of truth.

Every artifact the system produces is either stored inside this object or
content-addressed by hash and referenced from it (invariant #9). `Case` is
the only mutable aggregate; its components are immutable records.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from datamodel.audit import AuditActor, AuditEvent
from datamodel.clinical import CranialClinicalRecord
from datamodel.design import Design
from datamodel.scan_record import ScanRecord


class Case(BaseModel):
    """Top-level auditable unit of work.

    `clinical_record` is typed as the concrete `CranialClinicalRecord` (not
    the base `ClinicalRecord`) so the lossless round-trip guarantee
    (invariant #10) holds — see datamodel/clinical.py.
    """

    model_config = ConfigDict(validate_assignment=True)

    case_id: UUID = Field(default_factory=uuid4)
    device_class: str
    clinical_narrative: str
    clinical_record: CranialClinicalRecord
    scan_record: ScanRecord
    design_versions: list[Design] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def latest_design(self) -> Design | None:
        """The highest-version design, or None if there are none yet."""
        if not self.design_versions:
            return None
        return max(self.design_versions, key=lambda d: d.version)

    def append_audit(self, event: AuditEvent) -> None:
        """Append an audit event and bump `updated_at`."""
        self.audit_events = [*self.audit_events, event]
        self.updated_at = datetime.utcnow()

    def add_design_version(self, design: Design) -> None:
        """Append a new design version with correct `version` /
        `parent_version`, bump `updated_at`, and record an audit event.

        `Design` is immutable, so the version/parent are set on a copy.
        """
        prior = self.latest_design()
        next_version = 1 if prior is None else prior.version + 1
        parent_version = None if prior is None else prior.version
        versioned = design.model_copy(
            update={"version": next_version, "parent_version": parent_version}
        )
        self.design_versions = [*self.design_versions, versioned]
        self.updated_at = datetime.utcnow()
        self.audit_events = [
            *self.audit_events,
            AuditEvent(
                actor=AuditActor.SYSTEM,
                action="add_design_version",
                target_id=self.case_id,
                payload={"version": next_version, "parent_version": parent_version},
                rationale=design.change_summary,
            ),
        ]
