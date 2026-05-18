"""Tests for deriving canonical landmarks from synthetic-head params.

Covers determinism, schema completeness, on-surface anchoring, and that the
analytic+snap pipeline correctly reflects the parametric head's dimensions
and its occipital / brachycephaly deformations.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from anatomy.config import LandmarkName
from anatomy.landmarks import LandmarkSet
from anatomy.synthetic import derive_landmarks_from_synthetic
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head


@pytest.fixture
def baseline_config() -> HeadConfig:
    return HeadConfig()


@pytest.fixture
def plagio_config() -> HeadConfig:
    return HeadConfig(
        occipital_flat_mm=15.0,
        occipital_flat_radius_mm=55.0,
        occipital_flat_lateral_offset_mm=35.0,
    )


@pytest.fixture
def brachy_config() -> HeadConfig:
    return HeadConfig(
        occipital_flat_mm=6.0,
        occipital_flat_radius_mm=55.0,
        frontal_bossing_mm=5.0,
        frontal_bossing_radius_mm=40.0,
        brachycephaly_factor=0.85,
    )


def _pos(lset: LandmarkSet, name: LandmarkName) -> np.ndarray:
    return np.asarray(lset.landmarks[name].position_mm, dtype=np.float64)


def _dist(lset: LandmarkSet, a: LandmarkName, b: LandmarkName) -> float:
    return float(np.linalg.norm(_pos(lset, a) - _pos(lset, b)))


def test_derive_is_deterministic(
    baseline_config: HeadConfig,
    plagio_config: HeadConfig,
    brachy_config: HeadConfig,
) -> None:
    for cfg in (baseline_config, plagio_config, brachy_config):
        first = derive_landmarks_from_synthetic(cfg)
        second = derive_landmarks_from_synthetic(cfg)
        assert first == second
        assert first.to_json() == second.to_json()


def test_all_twelve_canonical_derived(baseline_config: HeadConfig) -> None:
    result = derive_landmarks_from_synthetic(baseline_config)
    assert result.frame == "canonical"
    assert len(result.landmarks) == 12
    assert set(result.landmarks.keys()) == set(LandmarkName)
    for lm in result.landmarks.values():
        assert lm.method == "derived"
        assert lm.confidence == 1.0


def test_landmarks_on_mesh_surface(
    baseline_config: HeadConfig,
    plagio_config: HeadConfig,
    brachy_config: HeadConfig,
) -> None:
    for cfg in (baseline_config, plagio_config, brachy_config):
        result = derive_landmarks_from_synthetic(cfg)
        mesh = generate_test_head(cfg)
        points = np.array(
            [result.landmarks[n].position_mm for n in LandmarkName],
            dtype=np.float64,
        )
        _closest, distances, _faces = trimesh.proximity.closest_point(mesh, points)
        assert np.max(distances) < 1e-3


def test_baseline_width_and_length(baseline_config: HeadConfig) -> None:
    result = derive_landmarks_from_synthetic(baseline_config)
    width = _dist(result, LandmarkName.EURYON_RIGHT, LandmarkName.EURYON_LEFT)
    length = _dist(result, LandmarkName.GLABELLA, LandmarkName.OPISTHOCRANION)
    assert width == pytest.approx(130.0, rel=0.01)
    assert length == pytest.approx(165.0, rel=0.01)


def test_brachy_shortens_ap_length(
    baseline_config: HeadConfig,
    brachy_config: HeadConfig,
) -> None:
    baseline = derive_landmarks_from_synthetic(baseline_config)
    brachy = derive_landmarks_from_synthetic(brachy_config)
    base_len = _dist(baseline, LandmarkName.GLABELLA, LandmarkName.OPISTHOCRANION)
    brachy_len = _dist(brachy, LandmarkName.GLABELLA, LandmarkName.OPISTHOCRANION)
    ratio = brachy_len / base_len
    assert 0.80 <= ratio <= 0.90


def test_tragion_symmetry_baseline(baseline_config: HeadConfig) -> None:
    result = derive_landmarks_from_synthetic(baseline_config)
    left = _pos(result, LandmarkName.TRAGION_LEFT)
    right = _pos(result, LandmarkName.TRAGION_RIGHT)
    assert left[0] == pytest.approx(-right[0], abs=1e-6)
    assert abs(left[1]) < 1e-6
    assert abs(right[1]) < 1e-6
    height_mm = baseline_config.height_mm
    for z in (left[2], right[2]):
        assert z > 0.0
        assert z < height_mm


def test_plagio_asymmetry_present(
    baseline_config: HeadConfig,
    plagio_config: HeadConfig,
) -> None:
    baseline = derive_landmarks_from_synthetic(baseline_config)
    plagio = derive_landmarks_from_synthetic(plagio_config)
    base_len = _dist(baseline, LandmarkName.GLABELLA, LandmarkName.OPISTHOCRANION)
    plagio_len = _dist(plagio, LandmarkName.GLABELLA, LandmarkName.OPISTHOCRANION)
    assert abs(plagio_len - base_len) > 1.0
