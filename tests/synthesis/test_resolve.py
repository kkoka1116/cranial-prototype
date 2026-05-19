"""Anchor resolution: correct patient-frame coords, loud failure, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from anatomy.config import LandmarkName
from anatomy.synthetic import derive_landmarks_from_synthetic
from datamodel.provenance import Provenance, ProvenanceSource, TracedValue
from datamodel.semantic import CorrectionZone, StrapMount, TrimLine, VentRegion
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head
from synthesis.errors import AnchorResolutionError
from synthesis.resolve import resolve_anchors

_CFG = HeadConfig(
    occipital_flat_mm=15.0, occipital_flat_radius_mm=55.0, occipital_flat_lateral_offset_mm=35.0
)


def _p() -> Provenance:
    return Provenance(source=ProvenanceSource.CLINICAL_INPUT, rationale="fixture", confidence=0.9)


def _tv(v):
    return TracedValue(value=v, provenance=_p())


def _head_and_lms():
    return generate_test_head(_CFG), derive_landmarks_from_synthetic(_CFG)


def _zone(region_ref="opisthocranion", anchors=("opisthocranion",)) -> CorrectionZone:
    return CorrectionZone(
        anatomical_anchors=list(anchors),
        region_ref=region_ref,
        target_offset_mm=_tv(6.0),
        pressure_target_mmHg=_tv(12.0),
        schedule=_tv("progressive"),
        duration_weeks=_tv(12),
        transition_smoothing_mm=_tv(20.0),
        falloff_function=_tv("gaussian"),
    )


def test_zone_resolves_to_patient_landmark() -> None:
    head, lms = _head_and_lms()
    r = resolve_anchors(_zone(), lms, head)
    np.testing.assert_allclose(
        r.center_xyz, lms.landmarks[LandmarkName.OPISTHOCRANION].as_array(), atol=1e-9
    )
    assert r.radius_mm == pytest.approx(20.0)


def test_zone_centroid_of_anchors_when_region_is_not_a_landmark() -> None:
    head, lms = _head_and_lms()
    z = _zone(region_ref="right_posterior_quadrant", anchors=("opisthocranion", "euryon_right"))
    r = resolve_anchors(z, lms, head)
    expected = np.mean(
        [
            lms.landmarks[LandmarkName.OPISTHOCRANION].as_array(),
            lms.landmarks[LandmarkName.EURYON_RIGHT].as_array(),
        ],
        axis=0,
    )
    np.testing.assert_allclose(r.center_xyz, expected, atol=1e-9)


def test_missing_landmark_raises_naming_it() -> None:
    head, lms = _head_and_lms()
    z = _zone(region_ref="not_a_landmark", anchors=("glabella", "made_up_point"))
    with pytest.raises(AnchorResolutionError, match="made_up_point"):
        resolve_anchors(z, lms, head)


def test_no_anchors_and_non_landmark_region_raises() -> None:
    head, lms = _head_and_lms()
    z = _zone(region_ref="some_region", anchors=())
    with pytest.raises(AnchorResolutionError, match="cannot resolve a center"):
        resolve_anchors(z, lms, head)


def test_resolution_is_deterministic() -> None:
    head, lms = _head_and_lms()
    z = _zone(region_ref="r", anchors=("opisthocranion", "euryon_right", "vertex"))
    assert resolve_anchors(z, lms, head) == resolve_anchors(z, lms, head)


def test_trimline_resolves_and_snaps() -> None:
    head, lms = _head_and_lms()
    tl = TrimLine(
        landmark_path=["glabella", "tragion_right", "opisthocranion", "tragion_left"],
        height_offsets_mm=_tv([2.0, -1.0, -3.0, -1.0]),
        edge_treatment=_tv("rolled"),
        edge_thickness_mm=_tv(2.5),
    )
    r = resolve_anchors(tl, lms, head)
    assert r.path_points_xyz is not None and len(r.path_points_xyz) == 4
    import trimesh

    pts = np.array(r.path_points_xyz)
    _, dist, _ = trimesh.proximity.closest_point(head, pts)
    assert float(np.max(dist)) < 1e-3  # snapped to surface


def test_trimline_length_mismatch_raises() -> None:
    head, lms = _head_and_lms()
    tl = TrimLine(
        landmark_path=["glabella", "opisthocranion", "tragion_left", "tragion_right"],
        height_offsets_mm=_tv([0.0, 0.0]),
        edge_treatment=_tv("straight"),
        edge_thickness_mm=_tv(2.0),
    )
    with pytest.raises(AnchorResolutionError, match="must match"):
        resolve_anchors(tl, lms, head)


def test_trimline_too_few_points_raises() -> None:
    head, lms = _head_and_lms()
    tl = TrimLine(
        landmark_path=["glabella", "opisthocranion"],
        height_offsets_mm=_tv([0.0, 0.0]),
        edge_treatment=_tv("straight"),
        edge_thickness_mm=_tv(2.0),
    )
    with pytest.raises(AnchorResolutionError, match=">= 4"):
        resolve_anchors(tl, lms, head)


def test_vent_and_strap_resolve() -> None:
    head, lms = _head_and_lms()
    vent = VentRegion(
        anatomical_anchors=["vertex"],
        region_ref="vertex",
        pattern=_tv("hex_lattice"),
        open_area_fraction=_tv(0.18),
        feature_size_mm=_tv(6.0),
    )
    rv = resolve_anchors(vent, lms, head)
    assert rv.center_xyz is not None and rv.extent_mm == pytest.approx(18.0)
    strap = StrapMount(
        anatomical_anchors=["tragion_left"],
        position_ref="tragion_left",
        strap_width_mm=_tv(20.0),
        angle_deg=_tv(35.0),
        reinforcement_radius_mm=_tv(8.0),
    )
    rs = resolve_anchors(strap, lms, head)
    assert rs.anchor_xyz is not None
    # Strap anchor is the semantic landmark position (the executor snaps it
    # onto the trimmed helmet at strap-phase time, not here).
    np.testing.assert_allclose(
        rs.anchor_xyz, lms.landmarks[LandmarkName.TRAGION_LEFT].as_array(), atol=1e-9
    )
