"""Measurement tests against the synthetic-head oracle.

For each of the three Week-1 example head configs: generate the synthetic
head, derive its canonical landmarks, and assert the measurements land in
the clinically expected ranges. Synthetic heads are already in the
canonical frame, so registration is a no-op here (the full register path
is exercised in test_pipeline.py).
"""

from __future__ import annotations

import pytest

from anatomy.measurements import (
    all_measurements,
    cephalic_index,
    cvai,
    head_circumference,
    head_height,
    head_length,
    head_width,
)
from anatomy.synthetic import derive_landmarks_from_synthetic
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head

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


@pytest.fixture(scope="module")
def built() -> dict[str, tuple]:
    out = {}
    for name, cfg in _CONFIGS.items():
        mesh = generate_test_head(cfg)
        lms = derive_landmarks_from_synthetic(cfg)
        out[name] = (cfg, mesh, lms)
    return out


def test_baseline_width_length_exact(built) -> None:
    _cfg, _mesh, lms = built["baseline"]
    assert head_width(lms).value == pytest.approx(130.0, rel=0.02)
    assert head_length(lms).value == pytest.approx(165.0, rel=0.02)


def test_baseline_cephalic_index_normocephalic(built) -> None:
    _cfg, _mesh, lms = built["baseline"]
    ci = cephalic_index(lms).value
    # width 130 / length 165 * 100 = 78.8 -> normocephalic band.
    assert 78.0 <= ci <= 83.0, f"baseline CI {ci:.2f} outside 78-83"


def test_baseline_cvai_low(built) -> None:
    _cfg, mesh, lms = built["baseline"]
    res = cvai(lms, mesh)
    assert res.value < 3.5, f"baseline CVAI {res.value:.2f} should be < 3.5 (symmetric)"


def test_plagio_cvai_moderate(built) -> None:
    _cfg, mesh, lms = built["plagio"]
    res = cvai(lms, mesh)
    assert 7.0 <= res.value <= 9.0, f"plagio CVAI {res.value:.2f} outside 7-9"


def test_brachy_cephalic_index_high(built) -> None:
    _cfg, _mesh, lms = built["brachy"]
    ci = cephalic_index(lms).value
    assert ci > 90.0, f"brachy CI {ci:.2f} should be > 90"


def test_head_circumference_plausible(built) -> None:
    _cfg, mesh, lms = built["baseline"]
    circ = head_circumference(mesh, lms).value
    # Infant head circumference ~ 410-480 mm; synthetic baseline is in range.
    assert 380.0 <= circ <= 520.0, f"baseline circumference {circ:.1f} mm implausible"


def test_head_height_positive(built) -> None:
    _cfg, _mesh, lms = built["baseline"]
    h = head_height(lms).value
    assert h > 0.0


def test_all_measurements_panel(built) -> None:
    _cfg, mesh, lms = built["baseline"]
    panel = all_measurements(mesh, lms)
    assert set(panel) == {
        "head_length",
        "head_width",
        "head_height",
        "cephalic_index",
        "cvai",
        "head_circumference",
    }
    assert all(r.quality in {"high", "medium", "low"} for r in panel.values())


def test_measurements_reject_source_frame(built) -> None:
    _cfg, _mesh, lms = built["baseline"]
    source_like = lms.model_copy(update={"frame": "source"})
    from anatomy.errors import MeasurementError

    with pytest.raises(MeasurementError, match="canonical"):
        head_length(source_like)
