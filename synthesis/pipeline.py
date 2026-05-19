"""Top-level synthesis pipeline: Design + ScanRecord -> patient helmet.

resolve -> translate (per semantic object, in list order) -> execute
(phase-structured) -> store helmet blob. Returns a `SynthesisResult` the
caller assembles into a `Design` version and persists via the Week 3
`CaseRepository`.

`synthesize` is deterministic: identical Design + ScanRecord -> identical
`helmet_hash` (kernel ops are deterministic + canonicalized; the blob hash
is the SHA-256 of the STL bytes). `op_id`s default to uuid4 and are NOT part
of the geometry or the hash.
"""

from __future__ import annotations

import io

import trimesh
from pydantic import BaseModel, ConfigDict

from datamodel.audit import AuditActor, AuditEvent
from datamodel.constraints import Constraint
from datamodel.design import Design
from datamodel.operations import KernelOperation
from datamodel.scan_record import ScanRecord
from storage.repository import CaseRepository
from synthesis.executor import execute
from synthesis.resolve import resolve_anchors
from synthesis.translator import translate


class SynthesisResult(BaseModel):
    """Auditable output of one synthesis run. The caller folds these into a
    `Design` (operations, constraints_checked, output_artifacts) and appends
    the design version to the `Case`."""

    model_config = ConfigDict(frozen=True)

    operations: list[KernelOperation]
    constraints: list[Constraint]
    helmet_hash: str
    audit_events: list[AuditEvent]


def synthesize(design: Design, scan_record: ScanRecord, repo: CaseRepository) -> SynthesisResult:
    """Resolve + translate every semantic object (in list order), execute the
    accumulated operations phase-structured, store the helmet, and return the
    full auditable result."""
    mesh_bytes = repo.load_blob(scan_record.mesh_hash)
    # STL is a triangle soup (no shared vertices); weld it back into a
    # manifold mesh so the kernel's per-vertex-normal offset is valid.
    # Deterministic for the pinned trimesh version, so helmet_hash stays
    # reproducible.
    patient_mesh = trimesh.load(file_obj=io.BytesIO(mesh_bytes), file_type="stl", process=True)
    patient_mesh.merge_vertices()

    operations: list[KernelOperation] = []
    audit: list[AuditEvent] = []

    for obj in design.semantic_objects:
        resolved = resolve_anchors(obj, scan_record.landmark_set, patient_mesh)
        these = translate(obj, resolved)
        operations.extend(these)
        audit.append(
            AuditEvent(
                actor=AuditActor.SYSTEM,
                action="resolve_translate",
                target_id=obj.id,
                payload={
                    "object_type": obj.type,
                    "op_ids": [str(o.op_id) for o in these],
                    "op_types": [o.op_type for o in these],
                },
                rationale=(
                    f"resolved {obj.type} anchors against the patient's "
                    f"LandmarkSet and translated to {len(these)} kernel "
                    f"operation(s)"
                ),
            )
        )

    result = execute(operations, patient_mesh)
    audit.extend(result.audit_events)

    helmet_hash = repo.save_blob(result.mesh.export(file_type="stl"), "model/stl")
    audit.append(
        AuditEvent(
            actor=AuditActor.SYSTEM,
            action="helmet_stored",
            payload={"helmet_hash": helmet_hash},
            rationale="synthesized helmet STL stored as a content-addressed blob",
        )
    )

    return SynthesisResult(
        operations=operations,
        constraints=result.constraints,
        helmet_hash=helmet_hash,
        audit_events=audit,
    )
