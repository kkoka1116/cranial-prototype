"""Scan loading and automatic scale correction.

`load_scan` is the single entry point. It detects the format by extension,
delegates mesh parsing to trimesh, then infers the units from the bounding
box and converts to millimeters. Every scale correction is logged and the
provenance is captured in a ScanMetadata model.

Invariant #3 (units in mm everywhere): a scan never leaves this module in
anything other than millimeters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from anatomy.config import (
    INFANT_LONGEST_MM_MAX,
    INFANT_LONGEST_MM_MIN,
    ScanMetadata,
)
from anatomy.errors import AnatomyError

logger = logging.getLogger("anatomy")

_SUPPORTED: dict[str, Literal["stl", "ply", "obj"]] = {
    ".stl": "stl",
    ".ply": "ply",
    ".obj": "obj",
}

# Conversion factors from a candidate unit into millimeters, with the
# plausible longest-bbox range (in that unit's native magnitude) that would
# imply the scan is in that unit. Ordered mm -> m -> inch.
_M_MIN = INFANT_LONGEST_MM_MIN / 1000.0  # 0.140
_M_MAX = INFANT_LONGEST_MM_MAX / 1000.0  # 0.200
_IN_MIN = INFANT_LONGEST_MM_MIN / 25.4  # ~5.51
_IN_MAX = INFANT_LONGEST_MM_MAX / 25.4  # ~7.87


def infer_and_correct_scale(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, Literal["mm", "m", "inch"], float]:
    """Infer the scan's units from its longest bbox dimension and convert to mm.

    Returns (mesh_in_mm, detected_units, scale_factor_applied). Raises
    AnatomyError if the longest dimension does not fall in any plausible
    infant-head range for mm, m, or inch.
    """
    extents = np.asarray(mesh.extents, dtype=np.float64)
    longest = float(np.max(extents))

    if INFANT_LONGEST_MM_MIN <= longest <= INFANT_LONGEST_MM_MAX:
        units: Literal["mm", "m", "inch"] = "mm"
        factor = 1.0
    elif _M_MIN <= longest <= _M_MAX:
        units = "m"
        factor = 1000.0
    elif _IN_MIN <= longest <= _IN_MAX:
        units = "inch"
        factor = 25.4
    else:
        raise AnatomyError(
            "Cannot infer scan scale: longest bounding-box dimension "
            f"{longest:.4f} is not a plausible infant head in mm "
            f"[{INFANT_LONGEST_MM_MIN}-{INFANT_LONGEST_MM_MAX}], "
            f"m [{_M_MIN:.3f}-{_M_MAX:.3f}], or inch "
            f"[{_IN_MIN:.2f}-{_IN_MAX:.2f}]. "
            f"Observed extents (raw units): {extents.tolist()}."
        )

    if factor != 1.0:
        scaled = mesh.copy()
        scaled.apply_scale(factor)
        logger.info(
            "scan scale corrected: %s -> mm (x%.4g); longest %.4f -> %.2f mm",
            units,
            factor,
            longest,
            longest * factor,
            extra={
                "anatomy_event": {
                    "operation": "infer_and_correct_scale",
                    "detected_units": units,
                    "scale_factor": factor,
                    "longest_before": longest,
                    "longest_after_mm": longest * factor,
                }
            },
        )
        return scaled, units, factor

    logger.info(
        "scan already in mm; longest %.2f mm (no scale correction)",
        longest,
        extra={
            "anatomy_event": {
                "operation": "infer_and_correct_scale",
                "detected_units": "mm",
                "scale_factor": 1.0,
                "longest_after_mm": longest,
            }
        },
    )
    return mesh, units, factor


def load_scan(path: Path) -> trimesh.Trimesh:
    """Load a 3D scan (STL/PLY/OBJ) and return it as a trimesh.Trimesh in mm.

    The single entry point for scan ingestion. Format is detected by
    extension; scale is inferred and corrected; provenance is logged as a
    ScanMetadata event. Raises AnatomyError on unsupported format, unreadable
    file, or uninferable scale.
    """
    path = Path(path)
    if not path.is_file():
        raise AnatomyError(f"scan file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise AnatomyError(
            f"unsupported scan format {suffix!r}; supported: " + ", ".join(sorted(_SUPPORTED))
        )
    fmt = _SUPPORTED[suffix]

    try:
        loaded = trimesh.load(path, file_type=fmt, process=False, force="mesh")
    except Exception as exc:  # trimesh raises a variety of types
        raise AnatomyError(f"failed to parse {fmt.upper()} scan {path}: {exc}") from exc

    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.shape[0] == 0:
        raise AnatomyError(f"scan {path} did not parse to a non-empty triangle mesh")

    mesh, units, factor = infer_and_correct_scale(loaded)

    meta = ScanMetadata(
        source_path=str(path),
        source_format=fmt,
        original_units=units,
        scale_factor_applied=factor,
        n_vertices=int(len(mesh.vertices)),
        n_faces=int(len(mesh.faces)),
    )
    logger.info(
        "scan loaded: %s (%s, %d verts, %d faces)",
        path.name,
        units,
        meta.n_vertices,
        meta.n_faces,
        extra={"anatomy_event": {"operation": "load_scan", **meta.model_dump()}},
    )
    return mesh
