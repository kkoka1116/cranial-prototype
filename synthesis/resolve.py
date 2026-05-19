"""Anchor resolution: semantic anchor names -> concrete patient-frame
geometry, read from this patient's registered LandmarkSet.

Deterministic (anchor sets are sorted by name before centroiding) and fails
loudly per invariant #12: a missing landmark or geometrically insufficient
anchors raise `AnchorResolutionError` — never a default, never a guess.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict

from anatomy.config import LandmarkName
from anatomy.landmarks import LandmarkSet, snap_to_surface
from datamodel.semantic import (
    CorrectionZone,
    ReliefZone,
    SemanticObject,
    StrapMount,
    TrimLine,
    VentRegion,
)
from synthesis.errors import AnchorResolutionError


class ResolvedGeometry(BaseModel):
    """Concrete patient-frame geometry an object needs to be translated.

    Only the fields relevant to the object's type are populated; the rest
    stay None. Frozen so a resolution result can't be mutated downstream.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    center_xyz: tuple[float, float, float] | None = None
    radius_mm: float | None = None
    path_points_xyz: list[tuple[float, float, float]] | None = None
    extent_mm: float | None = None
    anchor_xyz: tuple[float, float, float] | None = None


def _landmark_position(name: str, landmark_set: LandmarkSet, obj: SemanticObject) -> np.ndarray:
    """Position of a landmark by name, or AnchorResolutionError."""
    try:
        key = LandmarkName(name)
    except ValueError as exc:
        raise AnchorResolutionError(
            f"{type(obj).__name__} {obj.id}: anchor {name!r} is not a known "
            f"cranial landmark name"
        ) from exc
    if key not in landmark_set.landmarks:
        raise AnchorResolutionError(
            f"{type(obj).__name__} {obj.id}: landmark {name!r} is absent from "
            f"the patient's LandmarkSet (have: "
            f"{sorted(k.value for k in landmark_set.landmarks)})"
        )
    return landmark_set.landmarks[key].as_array()


def _is_landmark(name: str) -> bool:
    try:
        LandmarkName(name)
        return True
    except ValueError:
        return False


def _resolve_center(
    obj: SemanticObject, region_ref: str, landmark_set: LandmarkSet
) -> tuple[np.ndarray, bool]:
    """Center of a zone, plus whether it is a single named landmark.

    Returns (center_xyz, is_single_landmark). If `region_ref` names a
    landmark -> that landmark's position (is_single_landmark=True). Else the
    centroid of `anatomical_anchors` (sorted by name for determinism;
    is_single_landmark=False). Insufficient anchors -> AnchorResolutionError.
    """
    if _is_landmark(region_ref):
        return _landmark_position(region_ref, landmark_set, obj), True

    anchors = sorted(obj.anatomical_anchors)
    if not anchors:
        raise AnchorResolutionError(
            f"{type(obj).__name__} {obj.id}: region_ref {region_ref!r} is not a "
            f"landmark and there are no anatomical_anchors to take a centroid "
            f"of — cannot resolve a center"
        )
    pts = np.array([_landmark_position(a, landmark_set, obj) for a in anchors], dtype=np.float64)
    return pts.mean(axis=0), False


def _zone_radius_mm(
    obj: CorrectionZone | ReliefZone, landmark_set: LandmarkSet, *, is_single_landmark: bool
) -> float:
    """Effective radius. When the center is a single named landmark (a
    point) the radius is exactly `transition_smoothing_mm`. Only when the
    center is the centroid of a multi-anchor *region* is it widened to at
    least half the anchor spread so the whole region is covered. Widening a
    point-anchored zone by an unrelated anchor's distance would balloon the
    radius and self-intersect the naive offset. Deterministic."""
    base = float(obj.transition_smoothing_mm.value)
    if is_single_landmark:
        return base
    anchors = sorted(obj.anatomical_anchors)
    if len(anchors) >= 2:
        pts = np.array(
            [_landmark_position(a, landmark_set, obj) for a in anchors],
            dtype=np.float64,
        )
        spread = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                spread = max(spread, float(np.linalg.norm(pts[i] - pts[j])))
        return max(base, spread / 2.0)
    return base


def resolve_anchors(
    obj: SemanticObject, landmark_set: LandmarkSet, mesh: trimesh.Trimesh
) -> ResolvedGeometry:
    """Resolve a semantic object's anchors to concrete patient-frame
    geometry. Raises AnchorResolutionError (never degrades) per invariant #12.

    Currently implemented: CorrectionZone, ReliefZone (center + radius).
    TrimLine / VentRegion / StrapMount are added in the translator stage.
    """
    if isinstance(obj, CorrectionZone | ReliefZone):
        center, is_lm = _resolve_center(obj, obj.region_ref, landmark_set)
        radius = _zone_radius_mm(obj, landmark_set, is_single_landmark=is_lm)
        return ResolvedGeometry(
            kind=obj.type,
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
            radius_mm=radius,
        )

    if isinstance(obj, TrimLine):
        path = obj.landmark_path
        offsets = list(obj.height_offsets_mm.value)
        if len(path) != len(offsets):
            raise AnchorResolutionError(
                f"TrimLine {obj.id}: landmark_path has {len(path)} points but "
                f"height_offsets_mm has {len(offsets)} — must match"
            )
        if len(path) < 4:
            raise AnchorResolutionError(
                f"TrimLine {obj.id}: needs >= 4 path landmarks to define a "
                f"closed trim curve, got {len(path)}"
            )
        pts: list[tuple[float, float, float]] = []
        for name, dz in zip(path, offsets, strict=True):
            base = _landmark_position(name, landmark_set, obj)
            lifted = base + np.array([0.0, 0.0, float(dz)], dtype=np.float64)
            snapped = snap_to_surface(lifted, mesh).position_mm
            pts.append((float(snapped[0]), float(snapped[1]), float(snapped[2])))
        return ResolvedGeometry(kind=obj.type, path_points_xyz=pts)

    if isinstance(obj, VentRegion):
        center, _is_lm = _resolve_center(obj, obj.region_ref, landmark_set)
        # Patch extent: half the anchor spread if multi-anchor, else a
        # deterministic multiple of the vent feature size.
        anchors = sorted(obj.anatomical_anchors)
        if len(anchors) >= 2:
            apts = np.array(
                [_landmark_position(a, landmark_set, obj) for a in anchors],
                dtype=np.float64,
            )
            spread = max(
                float(np.linalg.norm(apts[i] - apts[j]))
                for i in range(len(apts))
                for j in range(i + 1, len(apts))
            )
            extent = max(spread / 2.0, 3.0 * float(obj.feature_size_mm.value))
        else:
            extent = 3.0 * float(obj.feature_size_mm.value)
        return ResolvedGeometry(
            kind=obj.type,
            center_xyz=(float(center[0]), float(center[1]), float(center[2])),
            extent_mm=extent,
        )

    if isinstance(obj, StrapMount):
        # The strap mounts on the HELMET trim edge, which does not exist at
        # resolve time. Resolve the semantic anchor (landmark position); the
        # executor's strap phase snaps it onto the actual trimmed helmet.
        anchor, _is_lm = _resolve_center(obj, obj.position_ref, landmark_set)
        return ResolvedGeometry(
            kind=obj.type,
            anchor_xyz=(float(anchor[0]), float(anchor[1]), float(anchor[2])),
        )

    raise AnchorResolutionError(
        f"resolve_anchors does not handle {type(obj).__name__} (object {obj.id})"
    )
