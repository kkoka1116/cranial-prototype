"""Anatomical measurements.

Every function here assumes the mesh and landmarks are already in the
canonical reference frame (invariant #7): origin between the ear canals at
skull base, +X patient's right, +Y anterior, +Z superior. Running these on
source-frame data is a bug — register first.

Measurements:
    cephalic_index   width / length * 100               (index)
    cvai             cranial vault asymmetry index       (percent)
    head_circumference  max horizontal slice perimeter   (mm)
    head_length      ||glabella - opisthocranion||        (mm)
    head_width       ||euryon_left - euryon_right||       (mm)
    head_height      vertex.z - mean(tragion z)           (mm)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict

from anatomy.config import LandmarkName
from anatomy.errors import MeasurementError
from anatomy.landmarks import LandmarkSet

logger = logging.getLogger("anatomy")

# CVAI diagonals are cast at this angle off the AP (anterior-posterior) axis.
CVAI_DIAGONAL_DEG: float = 30.0


class MeasurementResult(BaseModel):
    """One computed measurement plus provenance and a quality flag."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    unit: Literal["mm", "index", "percent"]
    inputs_used: list[str]
    quality: Literal["high", "medium", "low"]
    notes: str = ""


def _require_canonical(landmarks: LandmarkSet) -> None:
    if landmarks.frame != "canonical":
        raise MeasurementError(
            f"measurements require a canonical-frame LandmarkSet, got frame="
            f"{landmarks.frame!r}; register the scan first (invariant #7)"
        )


def _pos(landmarks: LandmarkSet, name: LandmarkName) -> np.ndarray:
    if name not in landmarks.landmarks:
        raise MeasurementError(f"required landmark missing: {name.value}")
    return landmarks.landmarks[name].as_array()


def head_length(landmarks: LandmarkSet) -> MeasurementResult:
    """AP length: straight distance glabella -> opisthocranion."""
    _require_canonical(landmarks)
    g = _pos(landmarks, LandmarkName.GLABELLA)
    o = _pos(landmarks, LandmarkName.OPISTHOCRANION)
    val = float(np.linalg.norm(g - o))
    return MeasurementResult(
        name="head_length",
        value=val,
        unit="mm",
        inputs_used=["glabella", "opisthocranion"],
        quality="high",
        notes="straight inter-landmark AP distance",
    )


def head_width(landmarks: LandmarkSet) -> MeasurementResult:
    """ML width: straight distance euryon_left -> euryon_right."""
    _require_canonical(landmarks)
    el = _pos(landmarks, LandmarkName.EURYON_LEFT)
    er = _pos(landmarks, LandmarkName.EURYON_RIGHT)
    val = float(np.linalg.norm(el - er))
    return MeasurementResult(
        name="head_width",
        value=val,
        unit="mm",
        inputs_used=["euryon_left", "euryon_right"],
        quality="high",
        notes="straight inter-landmark ML distance",
    )


def head_height(landmarks: LandmarkSet) -> MeasurementResult:
    """Vertical distance from vertex down to the tragion (ear-canal) plane."""
    _require_canonical(landmarks)
    v = _pos(landmarks, LandmarkName.VERTEX)
    tl = _pos(landmarks, LandmarkName.TRAGION_LEFT)
    tr = _pos(landmarks, LandmarkName.TRAGION_RIGHT)
    tragion_z = 0.5 * (float(tl[2]) + float(tr[2]))
    val = float(v[2]) - tragion_z
    return MeasurementResult(
        name="head_height",
        value=val,
        unit="mm",
        inputs_used=["vertex", "tragion_left", "tragion_right"],
        quality="high",
        notes="vertex.z minus mean tragion z (canonical frame)",
    )


def cephalic_index(landmarks: LandmarkSet) -> MeasurementResult:
    """Cephalic index = width / length * 100.

    < 76 dolichocephalic, 76-80.9 mesocephalic, 81-85.4 brachycephalic
    (clinical bands vary; we just report the number).
    """
    _require_canonical(landmarks)
    width = head_width(landmarks).value
    length = head_length(landmarks).value
    if length <= 0:
        raise MeasurementError("head length is non-positive; cannot form cephalic index")
    val = width / length * 100.0
    return MeasurementResult(
        name="cephalic_index",
        value=val,
        unit="index",
        inputs_used=["euryon_left", "euryon_right", "glabella", "opisthocranion"],
        quality="high",
        notes=f"width {width:.2f} / length {length:.2f} * 100",
    )


def _max_perimeter_slice(
    mesh: trimesh.Trimesh, z_lo: float, z_hi: float, n_steps: int = 25
) -> tuple[float, float]:
    """Search Z in [z_lo, z_hi] for the horizontal slice with the largest
    perimeter (the cranial-vault level). Returns (best_perimeter_mm, best_z).
    Shared by head_circumference and cvai so both reference the same
    clinically-standard "level of maximum head circumference".
    """
    best_perim = 0.0
    best_z = z_lo
    for z in np.linspace(z_lo, z_hi, n_steps):
        section = mesh.section(plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0])
        if section is None:
            continue
        try:
            planar, _ = section.to_planar()
        except Exception:  # degenerate slice
            continue
        perim = float(planar.length)
        if perim > best_perim:
            best_perim = perim
            best_z = float(z)
    return best_perim, best_z


def _diagonal_chord(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, str]:
    """Length of the mesh chord through `center` along ±`direction`.

    Returns (chord_length_mm, quality). Quality is "low" if either ray
    misses the mesh; "medium" if a ray has >1 forward hit (ambiguous);
    "high" otherwise.
    """
    direction = direction / np.linalg.norm(direction)
    origins = np.vstack([center, center])
    dirs = np.vstack([direction, -direction])
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True
    )
    if len(locations) == 0:
        return 0.0, "low"

    quality = "high"
    half = []
    for ray_idx in (0, 1):
        hits = locations[index_ray == ray_idx]
        if len(hits) == 0:
            return 0.0, "low"
        if len(hits) > 1:
            quality = "medium"
        dists = np.linalg.norm(hits - center, axis=1)
        half.append(float(np.max(dists)))  # outermost surface along this ray
    return half[0] + half[1], quality


def cvai(landmarks: LandmarkSet, mesh: trimesh.Trimesh) -> MeasurementResult:
    """Cranial Vault Asymmetry Index.

    Cast two diagonals through the geometric center of {glabella,
    opisthocranion, tragion_left, tragion_right} at ±30° from the AP axis,
    in the transverse (XY) plane. CVAI = |d1 - d2| / max(d1, d2) * 100.
    """
    _require_canonical(landmarks)
    g = _pos(landmarks, LandmarkName.GLABELLA)
    o = _pos(landmarks, LandmarkName.OPISTHOCRANION)
    tl = _pos(landmarks, LandmarkName.TRAGION_LEFT)
    tr = _pos(landmarks, LandmarkName.TRAGION_RIGHT)
    # In-plane (XY) center per the brief: centroid of the four registration
    # landmarks. CVAI is reported at the transverse level where the cranial
    # vault asymmetry is greatest, searched over the vault Z-range
    # [glabella height, just below the crown]. Rationale: clinical CVAI
    # quantifies the asymmetry at the most-affected level; using a single
    # fixed Z (e.g. the four-landmark mean, which the low tragion landmarks
    # drag down to ear-canal height) systematically misses occipital
    # plagiocephaly. A perfectly symmetric head yields ~0 at every level, so
    # the max-over-Z search cannot inflate a symmetric baseline.
    centroid = np.mean(np.vstack([g, o, tl, tr]), axis=0)
    z_lo = float(g[2])
    z_hi = float(mesh.bounds[1][2]) - 1.0
    if z_hi <= z_lo:
        z_hi = z_lo + 10.0

    ap = g - o
    ap[2] = 0.0  # project to transverse plane
    norm = np.linalg.norm(ap)
    if norm < 1e-9:
        raise MeasurementError("AP axis is degenerate in the transverse plane")
    ap = ap / norm

    theta = np.radians(CVAI_DIAGONAL_DEG)
    rot_pos = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_neg = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0.0],
            [np.sin(-theta), np.cos(-theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    diag_a = rot_pos @ ap
    diag_b = rot_neg @ ap

    best_val = -1.0
    best_a = best_b = 0.0
    best_z = z_lo
    quality = "low"
    for z in np.linspace(z_lo, z_hi, 30):
        center = np.array([centroid[0], centroid[1], float(z)], dtype=np.float64)
        len_a, q_a = _diagonal_chord(mesh, center, diag_a)
        len_b, q_b = _diagonal_chord(mesh, center, diag_b)
        if len_a <= 0.0 or len_b <= 0.0:
            continue
        longer = max(len_a, len_b)
        val = abs(len_a - len_b) / longer * 100.0
        if val > best_val:
            best_val = val
            best_a, best_b = len_a, len_b
            best_z = float(z)
            quality = (
                "low" if "low" in (q_a, q_b) else ("medium" if "medium" in (q_a, q_b) else "high")
            )

    if best_val < 0.0:
        raise MeasurementError("CVAI diagonal ray-cast found no mesh intersection")

    return MeasurementResult(
        name="cvai",
        value=best_val,
        unit="percent",
        inputs_used=["glabella", "opisthocranion", "tragion_left", "tragion_right", "mesh"],
        quality=quality,
        notes=f"peak at z={best_z:.1f} mm: diag(+30°)={best_a:.2f} mm, diag(-30°)={best_b:.2f} mm",
    )


def head_circumference(mesh: trimesh.Trimesh, landmarks: LandmarkSet) -> MeasurementResult:
    """Maximum horizontal-slice perimeter.

    Search Z from glabella height upward; the largest cross-section
    perimeter is the head circumference. Uses trimesh.section.
    """
    _require_canonical(landmarks)
    g = _pos(landmarks, LandmarkName.GLABELLA)
    z_lo = float(g[2])
    z_hi = float(mesh.bounds[1][2]) - 1.0  # a bit below the very top pole
    if z_hi <= z_lo:
        z_hi = z_lo + 10.0

    best_perim, best_z = _max_perimeter_slice(mesh, z_lo, z_hi)
    if best_perim <= 0.0:
        raise MeasurementError("could not find a valid horizontal slice for circumference")

    return MeasurementResult(
        name="head_circumference",
        value=best_perim,
        unit="mm",
        inputs_used=["glabella", "mesh"],
        quality="high",
        notes=f"max-perimeter slice at z={best_z:.1f} mm",
    )


def all_measurements(mesh: trimesh.Trimesh, landmarks: LandmarkSet) -> dict[str, MeasurementResult]:
    """Compute the full measurement panel on a registered mesh + landmarks."""
    _require_canonical(landmarks)
    results = {
        "head_length": head_length(landmarks),
        "head_width": head_width(landmarks),
        "head_height": head_height(landmarks),
        "cephalic_index": cephalic_index(landmarks),
        "cvai": cvai(landmarks, mesh),
        "head_circumference": head_circumference(mesh, landmarks),
    }
    logger.info(
        "measurements computed",
        extra={
            "anatomy_event": {
                "operation": "all_measurements",
                **{k: r.value for k, r in results.items()},
            }
        },
    )
    return results
