"""Tests for the Pydantic config layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kernel.config import (
    CorrectionRegion,
    HeadConfig,
    KernelConfig,
    ShellConfig,
    TrimConfig,
    TrimControlPoint,
    ValidationConfig,
)


def test_head_config_defaults_match_brief() -> None:
    cfg = HeadConfig()
    assert cfg.length_mm == 165.0
    assert cfg.width_mm == 130.0
    assert cfg.height_mm == 115.0


def test_head_config_rejects_negative_dimensions() -> None:
    with pytest.raises(ValidationError):
        HeadConfig(length_mm=-1.0)


def test_correction_region_rejects_unknown_falloff() -> None:
    with pytest.raises(ValidationError):
        CorrectionRegion(
            x_mm=0,
            y_mm=0,
            z_mm=0,
            radius_mm=10.0,
            magnitude_mm=1.0,
            falloff="cubic",  # noqa
        )


def test_shell_config_is_frozen() -> None:
    cfg = ShellConfig()
    with pytest.raises(ValidationError):
        cfg.relief_mm = 99.0  # type: ignore[misc]


def test_validation_config_defaults_match_brief() -> None:
    cfg = ValidationConfig()
    assert cfg.min_wall_thickness_mm == 3.0
    assert cfg.max_mass_g == 250.0
    assert cfg.material_density_g_per_cm3 == 1.27


def test_kernel_config_loads_from_yaml(tmp_path: Path) -> None:
    data = {
        "name": "test",
        "trim": {
            "control_points": [
                {"azimuth_deg": 0.0, "elevation_deg": -10.0},
                {"azimuth_deg": 90.0, "elevation_deg": -10.0},
                {"azimuth_deg": 180.0, "elevation_deg": -10.0},
                {"azimuth_deg": 270.0, "elevation_deg": -10.0},
            ],
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = KernelConfig.from_yaml(p)
    assert cfg.name == "test"
    assert isinstance(cfg.head, HeadConfig)
    assert isinstance(cfg.trim, TrimConfig)


def test_kernel_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        KernelConfig.from_yaml(p)


def test_kernel_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HeadConfig(unknown_field=42)  # type: ignore[call-arg]


def test_trim_min_4_control_points_required() -> None:
    with pytest.raises(ValidationError):
        TrimConfig(
            control_points=[
                TrimControlPoint(azimuth_deg=0.0, elevation_deg=0.0),
                TrimControlPoint(azimuth_deg=180.0, elevation_deg=0.0),
            ]
        )
