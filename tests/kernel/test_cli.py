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


def test_cli_exits_2_on_missing_config(tmp_path: Path) -> None:
    """Config-load failure returns exit code 2."""
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
    assert rc == 2


def test_cli_exits_3_on_wall_thickness_failure(tmp_path: Path) -> None:
    """A config that produces a too-thin wall returns exit code 3 (validation failure)."""
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
    assert rc == 3
    # Pipeline must NOT have written an STL on failure.
    assert not (tmp_path / "out.stl").exists()


def test_cli_exits_4_on_geometry_error(tmp_path: Path, monkeypatch) -> None:
    """Force a GeometryError mid-pipeline and confirm the CLI returns 4."""
    import kernel.generate as generate_mod
    from kernel.errors import GeometryError

    def boom(*_a, **_kw):
        raise GeometryError("synthetic geometry failure", context={"why": "test"})

    monkeypatch.setattr(generate_mod, "build_shell", boom)

    rc = main(
        [
            "--config",
            str(REPO_ROOT / "examples" / "baseline.yaml"),
            "--out",
            str(tmp_path / "out.stl"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 4


def test_cli_exits_5_on_generic_kernel_error(tmp_path: Path, monkeypatch) -> None:
    """A plain KernelError (neither validation nor geometry) returns 5."""
    import kernel.generate as generate_mod
    from kernel.errors import KernelError

    def boom(*_a, **_kw):
        raise KernelError("synthetic kernel failure")

    monkeypatch.setattr(generate_mod, "build_shell", boom)

    rc = main(
        [
            "--config",
            str(REPO_ROOT / "examples" / "baseline.yaml"),
            "--out",
            str(tmp_path / "out.stl"),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 5


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
