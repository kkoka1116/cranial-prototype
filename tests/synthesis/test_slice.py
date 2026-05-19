"""De-risking slice: one CorrectionZone, one landmark, one synthetic head.

resolve -> translate -> shell-phase execute -> a valid helmet shell whose
corrected region is demonstrably where the anchored landmark is, and which
is deterministic and carries semantic provenance. Proven before widening to
the full five-type pipeline.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from anatomy.config import LandmarkName
from anatomy.synthetic import derive_landmarks_from_synthetic
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.semantic import CorrectionZone
from kernel.config import HeadConfig, ShellConfig
from kernel.operations.shell import build_shell
from kernel.primitives.test_head import generate_test_head
from synthesis.executor import execute
from synthesis.resolve import resolve_anchors
from synthesis.translator import translate

_PLAGIO = HeadConfig(
    occipital_flat_mm=15.0,
    occipital_flat_radius_mm=55.0,
    occipital_flat_lateral_offset_mm=35.0,
)


def _prov() -> Provenance:
    return Provenance(
        source=ProvenanceSource.CLINICAL_INPUT,
        rationale="hand-authored slice fixture: correct the flattened right occiput",
        confidence=0.9,
    )


def _tv(v):
    return TracedValue(value=v, provenance=_prov())


def _zone() -> CorrectionZone:
    return CorrectionZone(
        anatomical_anchors=["opisthocranion"],
        region_ref="opisthocranion",
        target_offset_mm=_tv(2.5),
        pressure_target_mmHg=_tv(12.0),
        schedule=_tv("progressive"),
        duration_weeks=_tv(12),
        transition_smoothing_mm=_tv(22.0),
        falloff_function=_tv("gaussian"),
    )


def _stl_hash(mesh) -> str:
    return hashlib.sha256(mesh.export(file_type="stl")).hexdigest()


def test_slice_resolves_translates_executes_to_valid_shell() -> None:
    head = generate_test_head(_PLAGIO)
    lms = derive_landmarks_from_synthetic(_PLAGIO)
    zone = _zone()

    resolved = resolve_anchors(zone, lms, head)
    # Center resolved to the actual opisthocranion position for THIS patient.
    expected = lms.landmarks[LandmarkName.OPISTHOCRANION].as_array()
    np.testing.assert_allclose(resolved.center_xyz, expected, atol=1e-9)
    assert resolved.radius_mm == pytest.approx(22.0)

    ops = translate(zone, resolved)
    assert len(ops) == 1
    op = ops[0]
    assert op.op_type == "offset_surface"
    assert op.inputs["direction"] == "inward"
    assert op.inputs["magnitude_mm"] == pytest.approx(2.5)
    assert op.inputs["signed_magnitude_mm"] == pytest.approx(-2.5)
    # Invariant #11: every op traces to a semantic object.
    assert op.derived_from == [zone.id]

    shell = execute(ops, head).mesh
    assert shell.is_watertight
    assert shell.is_winding_consistent
    assert shell.is_volume


def test_slice_is_deterministic() -> None:
    head = generate_test_head(_PLAGIO)
    lms = derive_landmarks_from_synthetic(_PLAGIO)
    zone = _zone()

    r1 = resolve_anchors(zone, lms, head)
    r2 = resolve_anchors(zone, lms, head)
    assert r1 == r2  # resolution deterministic

    o1 = translate(zone, r1)[0]
    o2 = translate(zone, r2)[0]
    # Operation *content* deterministic (op_id is uuid4-defaulted, excluded).
    assert (o1.op_type, o1.inputs, o1.derived_from) == (
        o2.op_type,
        o2.inputs,
        o2.derived_from,
    )

    s1 = execute([o1], head).mesh
    s2 = execute([o2], head).mesh
    assert _stl_hash(s1) == _stl_hash(s2)


def test_correction_lands_at_the_anchored_landmark() -> None:
    """The corrected region is demonstrably where the landmark is: the
    largest deviation between the anchored shell's outer surface and a
    no-correction baseline outer surface falls within the zone radius of
    the resolved (opisthocranion) center — proving anchoring is functional,
    not cosmetic."""
    head = generate_test_head(_PLAGIO)
    lms = derive_landmarks_from_synthetic(_PLAGIO)
    zone = _zone()
    resolved = resolve_anchors(zone, lms, head)
    ops = translate(zone, resolved)

    corrected_shell = execute(ops, head).mesh
    baseline_shell, _bi, _bo = build_shell(head, ShellConfig())

    import trimesh

    samples, _ = trimesh.sample.sample_surface_even(corrected_shell, 4000, seed=0)
    _, dist, _ = trimesh.proximity.closest_point(baseline_shell, samples)
    worst = samples[int(np.argmax(dist))]
    center = np.asarray(resolved.center_xyz)
    assert float(np.max(dist)) > 1.0, "correction produced no measurable deviation"
    assert float(np.linalg.norm(worst - center)) < 1.5 * resolved.radius_mm, (
        "largest shell deviation is not near the anchored opisthocranion — "
        "anchoring is not functional"
    )
