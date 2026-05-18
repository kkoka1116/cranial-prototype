"""End-to-end Week-2 acceptance test.

For each of the three Week-1 example head configs, run the full pipeline:

    synthetic head + derived landmarks
      -> apply a KNOWN rigid transform (simulate a raw-scan pose)
      -> clean_scan
      -> register (recover canonical frame from the source landmarks)
      -> measurements

and assert the registered measurements round-trip the measurements taken
directly on the un-transformed canonical synthetic head within 2%
(acceptance #5). This simultaneously exercises acceptance #7 (a rotated +
translated head is re-registered correctly).

Also guards the committed examples/*_landmarks.json against synthetic.py
drift (acceptance #10) and JSON round-trip (acceptance #3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from anatomy.cleanup import clean_scan
from anatomy.config import LANDMARK_ORDER
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import all_measurements
from anatomy.registration import apply_transform, register
from anatomy.synthetic import derive_landmarks_from_synthetic
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = REPO_ROOT / "examples"

_CONFIGS = {
    "baseline": HeadConfig(),
    "plagio": HeadConfig(
        occipital_flat_mm=15.0,
        occipital_flat_radius_mm=55.0,
        occipital_flat_lateral_offset_mm=35.0,
    ),
    "brachy": HeadConfig(
        occipital_flat_mm=6.0,
        occipital_flat_radius_mm=55.0,
        frontal_bossing_mm=5.0,
        frontal_bossing_radius_mm=40.0,
        brachycephaly_factor=0.85,
    ),
}

_EXAMPLE_FILES = {
    "baseline": "baseline_landmarks.json",
    "plagio": "plagio_landmarks.json",
    "brachy": "brachy_landmarks.json",
}


def _known_transform() -> np.ndarray:
    """A fixed rotation (about a tilted axis) + translation, as a 4x4."""
    axis = np.array([0.3, -0.7, 0.65], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = math.radians(37.0)
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = axis
    rot = np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = np.array([41.0, -23.0, 88.0])
    return transform


def _transform_points(pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.column_stack([pts, np.ones(len(pts))])
    return (transform @ hom.T).T[:, :3]


def _source_set_from(lms_canon: LandmarkSet, transform: np.ndarray) -> LandmarkSet:
    pos = _transform_points(
        np.array([lms_canon.landmarks[n].position_mm for n in LANDMARK_ORDER]),
        transform,
    )
    return LandmarkSet(
        frame="source",
        landmarks={
            n: Landmark(name=n, position_mm=tuple(pos[i]), method="manual")
            for i, n in enumerate(LANDMARK_ORDER)
        },
    )


@pytest.fixture(scope="module", params=list(_CONFIGS))
def case(request):
    name = request.param
    cfg = _CONFIGS[name]
    mesh_canon = generate_test_head(cfg)
    lms_canon = derive_landmarks_from_synthetic(cfg)

    # The pipeline's canonical frame is the *registration* frame (anatomically
    # defined: +Y toward glabella). It differs from the kernel's geometric
    # ellipsoid frame by a small fixed rotation, so the round-trip reference
    # is the registered un-transformed head — not the raw kernel-frame mesh.
    # Registration is exactly equivariant (proven to ~1e-14 mm in
    # test_registration), so register(T·x) must reproduce register(x).
    src_identity = _source_set_from(lms_canon, np.eye(4))
    m0, lm0 = register(src_identity)
    mesh0 = apply_transform(mesh_canon, m0)
    reference = all_measurements(mesh0, lm0)
    return name, cfg, mesh_canon, lms_canon, reference


def test_full_pipeline_is_pose_invariant(case) -> None:
    """Acceptance #5 + #7: a head given in an arbitrary source pose, run
    through cleanup -> register -> measurements, reproduces the canonical
    measurements within 2% (the pipeline is invariant to source pose)."""
    name, _cfg, mesh_canon, lms_canon, reference = case
    transform = _known_transform()

    src_mesh = apply_transform(mesh_canon, transform)
    src_landmarks = _source_set_from(lms_canon, transform)

    cleaned = clean_scan(src_mesh)  # synthetic head already clean -> ~no-op
    matrix, canon_landmarks = register(src_landmarks)
    canon_mesh = apply_transform(cleaned.mesh, matrix)
    measured = all_measurements(canon_mesh, canon_landmarks)

    for key in ("cephalic_index", "head_length", "head_width", "head_circumference"):
        exp = reference[key].value
        got = measured[key].value
        assert got == pytest.approx(
            exp, rel=0.02
        ), f"[{name}] {key}: pose-variant — expected {exp:.3f}, got {got:.3f}"
    assert measured["cvai"].value == pytest.approx(
        reference["cvai"].value, rel=0.02, abs=0.5
    ), f"[{name}] cvai pose-variant: {reference['cvai'].value:.3f} vs {measured['cvai'].value:.3f}"


def test_input_parameters_round_trip(case) -> None:
    """Acceptance #5: the undeformed baseline recovers its HeadConfig input
    dimensions exactly; deformed configs legitimately change raw dimensions
    (that IS the deformation), so their round-trip target is the clinical
    index (CVAI/CI), asserted in test_clinical_ranges."""
    name, cfg, _m, _l, reference = case
    if name == "baseline":
        assert reference["head_width"].value == pytest.approx(cfg.width_mm, rel=0.02)
        assert reference["head_length"].value == pytest.approx(cfg.length_mm, rel=0.02)
    elif name == "brachy":
        # AP-compressed by brachycephaly_factor; width roughly preserved.
        assert reference["head_length"].value == pytest.approx(
            cfg.length_mm * cfg.brachycephaly_factor, rel=0.06
        )
        assert reference["head_width"].value == pytest.approx(cfg.width_mm, rel=0.04)
    else:  # plagio: asymmetric flattening narrows/shortens the head
        assert reference["head_width"].value < cfg.width_mm
        assert reference["head_length"].value < cfg.length_mm
        assert reference["head_width"].value == pytest.approx(cfg.width_mm, rel=0.05)


def test_clinical_ranges(case) -> None:
    name, _cfg, _m, _l, reference = case
    ci = reference["cephalic_index"].value
    cv = reference["cvai"].value
    circ = reference["head_circumference"].value
    assert 380.0 <= circ <= 520.0
    if name == "baseline":
        assert 78.0 <= ci <= 83.0
        assert cv < 3.5
    elif name == "plagio":
        assert 7.0 <= cv <= 9.0
    elif name == "brachy":
        assert ci > 90.0


def test_committed_example_json_round_trips_and_matches(case) -> None:
    name, cfg, _m, _l, _b = case
    path = EXAMPLES / _EXAMPLE_FILES[name]
    assert path.is_file(), f"missing committed example: {path}"
    loaded = LandmarkSet.from_json(path.read_text(encoding="utf-8"))
    assert loaded.frame == "canonical"
    # No drift: re-deriving must reproduce the committed file exactly.
    fresh = derive_landmarks_from_synthetic(cfg)
    assert loaded == fresh, f"{name}: committed landmarks JSON drifted from synthetic.py"
