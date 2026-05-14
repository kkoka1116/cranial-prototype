"""Smoke tests for the CLI in kernel.generate."""

from __future__ import annotations

from pathlib import Path

from kernel.config import HeadConfig, KernelConfig, ShellConfig, TrimConfig, TrimControlPoint
from kernel.generate import main, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cli_succeeds_on_baseline(tmp_path: Path) -> None:
    out = tmp_path / "helmet.stl"
    rc = main(
        [
            "--config",
            str(REPO_ROOT / "examples" / "baseline.yaml"),
            "--out",
            str(out),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert out.exists()
    # Non-empty STL.
    data = out.read_bytes()
    assert len(data) > 100


def test_cli_fails_on_missing_config(tmp_path: Path) -> None:
    rc = main(
        [
            "--config",
            str(tmp_path / "nope.yaml"),
            "--out",
            str(tmp_path / "helmet.stl"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc != 0


def test_cli_fails_on_invalid_geometry(tmp_path: Path) -> None:
    """A config that produces an invalid mesh exits non-zero.

    We force this by setting min_wall_thickness above the physical wall
    thickness, so validation fails.
    """
    import yaml

    cfg = {
        "name": "too-thin",
        "shell": {"wall_thickness_mm": 1.0, "relief_mm": 3.0},
        "trim": {
            "control_points": [
                {"azimuth_deg": 0.0, "elevation_deg": -10.0},
                {"azimuth_deg": 90.0, "elevation_deg": -10.0},
                {"azimuth_deg": 180.0, "elevation_deg": -10.0},
                {"azimuth_deg": 270.0, "elevation_deg": -10.0},
            ],
        },
        "validation": {
            "min_wall_thickness_mm": 3.0,
            "max_mass_g": 1000.0,
            "max_bbox_extent_mm": 1000.0,
        },
    }
    cfg_path = tmp_path / "thin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = main(
        [
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "out.stl"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc != 0


def test_run_pipeline_returns_helmet_and_results() -> None:
    cfg = KernelConfig(
        name="t",
        head=HeadConfig(subdivisions=3),
        shell=ShellConfig(),
        trim=TrimConfig(
            control_points=[
                TrimControlPoint(azimuth_deg=0.0, elevation_deg=-5.0),
                TrimControlPoint(azimuth_deg=90.0, elevation_deg=-5.0),
                TrimControlPoint(azimuth_deg=180.0, elevation_deg=-5.0),
                TrimControlPoint(azimuth_deg=270.0, elevation_deg=-5.0),
            ],
        ),
    )
    helmet, results = run_pipeline(cfg)
    assert helmet.is_volume
    assert len(results) == 5
    assert all(r.passed for r in results)
