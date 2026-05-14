"""CLI entry point: load config, run the kernel pipeline, write STL.

Usage:
    uv run python -m kernel.generate --config examples/baseline.yaml --out helmet.stl

The pipeline is:

    1. Load + validate config (Pydantic).
    2. Generate synthetic head mesh.
    3. Build helmet shell (outer/inner offset + boolean subtract).
    4. Apply trim line cut.
    5. Run validation suite (manifold, watertight, wall thickness, mass, bbox).
    6. Write STL to disk.

Every step emits a single structured JSON log line. The shape (operation name,
input parameter hash, output mesh hash, duration ms, status) forward-ports
cleanly to the week-3 AuditEvent without being the audit trail itself.

Failures (validation failure, geometry error, config error) exit with a
non-zero status code and a final JSON line describing the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import trimesh

from kernel.config import KernelConfig
from kernel.errors import GeometryError, KernelError, ValidationError
from kernel.export.stl import sha256_of_mesh_stl, write_stl_to_path
from kernel.operations.shell import build_shell
from kernel.operations.trim import apply_trim
from kernel.primitives.test_head import generate_test_head
from kernel.validation.geometry import (
    check_bbox_within,
    check_mass,
    check_min_wall_thickness,
)
from kernel.validation.topology import (
    ValidationResult,
    check_manifold,
    check_watertight,
)

logger = logging.getLogger("kernel")


# --------------------------------------------------------------------------- #
# Structured JSON logging                                                     #
# --------------------------------------------------------------------------- #


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "kernel_event", None)
        if extra is not None:
            payload["event"] = extra
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def _log_event(
    operation: str,
    *,
    input_hash: str,
    output_hash: str | None,
    duration_ms: float,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    event = {
        "operation": operation,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "duration_ms": round(duration_ms, 3),
        "status": status,
    }
    if extra:
        event.update(extra)
    logger.info(operation, extra={"kernel_event": event})


def _hash_mesh(mesh: trimesh.Trimesh) -> str:
    return sha256_of_mesh_stl(mesh)


def _hash_config_obj(obj: Any) -> str:
    """Stable hash of a Pydantic model or any JSON-serializable structure."""
    if hasattr(obj, "model_dump_json"):
        data = obj.model_dump_json()
    else:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #


def run_pipeline(config: KernelConfig) -> tuple[trimesh.Trimesh, list[ValidationResult]]:
    """Execute the full kernel pipeline and return (helmet_mesh, validation_results).

    Raises ValidationError if any validation check fails. Raises GeometryError
    if an internal geometric operation fails (e.g., empty boolean result).
    """
    config_hash = _hash_config_obj(config)

    # 1. Synthetic head ------------------------------------------------------
    t0 = time.perf_counter()
    head = generate_test_head(config.head)
    head_hash = _hash_mesh(head)
    _log_event(
        "generate_test_head",
        input_hash=_hash_config_obj(config.head),
        output_hash=head_hash,
        duration_ms=(time.perf_counter() - t0) * 1000,
        status="ok",
        extra={
            "n_vertices": int(len(head.vertices)),
            "n_faces": int(len(head.faces)),
        },
    )

    # 2. Shell ---------------------------------------------------------------
    t0 = time.perf_counter()
    shell, inner, outer = build_shell(head, config.shell)
    shell_hash = _hash_mesh(shell)
    _log_event(
        "build_shell",
        input_hash=_hash_config_obj(config.shell),
        output_hash=shell_hash,
        duration_ms=(time.perf_counter() - t0) * 1000,
        status="ok",
        extra={
            "n_vertices": int(len(shell.vertices)),
            "n_faces": int(len(shell.faces)),
        },
    )

    # 3. Trim ----------------------------------------------------------------
    t0 = time.perf_counter()
    helmet = apply_trim(shell, config.trim)
    helmet_hash = _hash_mesh(helmet)
    _log_event(
        "apply_trim",
        input_hash=_hash_config_obj(config.trim),
        output_hash=helmet_hash,
        duration_ms=(time.perf_counter() - t0) * 1000,
        status="ok",
        extra={
            "n_vertices": int(len(helmet.vertices)),
            "n_faces": int(len(helmet.faces)),
        },
    )

    # 4. Validation ----------------------------------------------------------
    t0 = time.perf_counter()
    results: list[ValidationResult] = []
    results.append(check_manifold(helmet))
    results.append(check_watertight(helmet))
    # Wall thickness is a property of the offset construction, not of the
    # trim. Measuring on the trimmed surfaces would read ~0 near the cut
    # edge, where inner and outer share boolean-introduced geometry. We
    # use the untrimmed inner/outer surfaces — the bulk wall thickness
    # is unchanged by trimming.
    results.append(
        check_min_wall_thickness(
            inner_surface_mesh=inner,
            outer_surface_mesh=outer,
            min_thickness_mm=config.validation.min_wall_thickness_mm,
        )
    )
    results.append(
        check_mass(
            helmet,
            density_g_per_cm3=config.validation.material_density_g_per_cm3,
            max_mass_g=config.validation.max_mass_g,
        )
    )
    results.append(check_bbox_within(helmet, max_extent_mm=config.validation.max_bbox_extent_mm))
    _log_event(
        "validate",
        input_hash=helmet_hash,
        output_hash=None,
        duration_ms=(time.perf_counter() - t0) * 1000,
        status="ok" if all(r.passed for r in results) else "failed",
        extra={
            "checks": [
                {
                    "metric": r.metric_name,
                    "passed": r.passed,
                    "actual": r.actual_value,
                    "limit": r.limit_value,
                }
                for r in results
            ]
        },
    )

    # Raise on the first failing check.
    for r in results:
        if not r.passed:
            raise ValidationError(r)

    _log_event(
        "pipeline",
        input_hash=config_hash,
        output_hash=helmet_hash,
        duration_ms=0.0,
        status="ok",
    )
    return helmet, results


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="kernel.generate",
        description="Generate a manifold + watertight cranial helmet STL from a config.",
    )
    p.add_argument("--config", required=True, type=Path, help="Path to YAML kernel config.")
    p.add_argument("--out", required=True, type=Path, help="Path to write the STL output.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(level=getattr(logging, args.log_level))

    try:
        config = KernelConfig.from_yaml(args.config)
    except Exception as exc:
        logger.error(
            "config_load_failed",
            extra={
                "kernel_event": {
                    "operation": "load_config",
                    "status": "error",
                    "error": str(exc),
                    "config_path": str(args.config),
                }
            },
        )
        return 2

    try:
        helmet, _results = run_pipeline(config)
    except ValidationError as exc:
        logger.error(
            "validation_failed",
            extra={
                "kernel_event": {
                    "operation": "validate",
                    "status": "failed",
                    "metric": exc.result.metric_name,
                    "actual": exc.result.actual_value,
                    "limit": exc.result.limit_value,
                    "message": exc.result.message,
                }
            },
        )
        return 3
    except GeometryError as exc:
        logger.error(
            "geometry_error",
            extra={
                "kernel_event": {
                    "operation": "geometry",
                    "status": "error",
                    "error": str(exc),
                    "context": exc.context,
                }
            },
        )
        return 4
    except KernelError as exc:
        logger.error(
            "kernel_error",
            extra={
                "kernel_event": {
                    "operation": "kernel",
                    "status": "error",
                    "error": str(exc),
                }
            },
        )
        return 5

    stl_hash = write_stl_to_path(helmet, args.out)
    _log_event(
        "write_stl",
        input_hash=_hash_mesh(helmet),
        output_hash=stl_hash,
        duration_ms=0.0,
        status="ok",
        extra={"path": str(args.out)},
    )
    # Also print the hash on stdout for easy capture / golden-test workflows.
    sys.stdout.write(stl_hash + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
