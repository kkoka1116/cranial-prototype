"""Shared anatomy enums, frame tags, and scan metadata models.

The landmark names here are the de-facto cranial-anthropometry standard and
are used verbatim (exact casing) in landmark JSON files. Keeping the enum
and frame literal here (rather than in landmarks.py) keeps the import graph
acyclic: landmarks.py / registration.py / measurements.py all depend on
config.py, never the reverse.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LandmarkName(str, Enum):
    """The twelve standard cranial landmarks. Order of definition matches the
    annotation-tool prompt order in LANDMARK_ORDER."""

    GLABELLA = "glabella"
    NASION = "nasion"
    VERTEX = "vertex"
    OPISTHOCRANION = "opisthocranion"
    TRAGION_LEFT = "tragion_left"
    TRAGION_RIGHT = "tragion_right"
    EURYON_LEFT = "euryon_left"
    EURYON_RIGHT = "euryon_right"
    FRONTOTEMPORALE_LEFT = "frontotemporale_left"
    FRONTOTEMPORALE_RIGHT = "frontotemporale_right"
    PARIETAL_EMINENCE_LEFT = "parietal_eminence_left"
    PARIETAL_EMINENCE_RIGHT = "parietal_eminence_right"


# Reference frame a set of landmark coordinates lives in. "source" = raw scan
# coordinates as loaded; "canonical" = registered (origin between ear canals
# at skull base, +X right, +Y anterior, +Z superior — see CLAUDE.md).
Frame = Literal["source", "canonical"]


# Prompt / iteration order (the brief's listed order). Tuple so iteration is
# deterministic.
LANDMARK_ORDER: tuple[LandmarkName, ...] = (
    LandmarkName.GLABELLA,
    LandmarkName.NASION,
    LandmarkName.VERTEX,
    LandmarkName.OPISTHOCRANION,
    LandmarkName.TRAGION_LEFT,
    LandmarkName.TRAGION_RIGHT,
    LandmarkName.EURYON_LEFT,
    LandmarkName.EURYON_RIGHT,
    LandmarkName.FRONTOTEMPORALE_LEFT,
    LandmarkName.FRONTOTEMPORALE_RIGHT,
    LandmarkName.PARIETAL_EMINENCE_LEFT,
    LandmarkName.PARIETAL_EMINENCE_RIGHT,
)

# The minimum set required to compute the canonical-frame transform.
REGISTRATION_LANDMARKS: tuple[LandmarkName, ...] = (
    LandmarkName.GLABELLA,
    LandmarkName.OPISTHOCRANION,
    LandmarkName.TRAGION_LEFT,
    LandmarkName.TRAGION_RIGHT,
)

# Human-readable one-line hint per landmark for the annotation tool.
LANDMARK_HINTS: dict[LandmarkName, str] = {
    LandmarkName.GLABELLA: "midline anterior, most prominent forehead point between brow ridges",
    LandmarkName.NASION: "midline depression between forehead and nasal root",
    LandmarkName.VERTEX: "topmost point on the skull in canonical frame",
    LandmarkName.OPISTHOCRANION: "most posterior midline point on the occiput",
    LandmarkName.TRAGION_LEFT: "superior notch of the LEFT ear canal",
    LandmarkName.TRAGION_RIGHT: "superior notch of the RIGHT ear canal",
    LandmarkName.EURYON_LEFT: "most lateral point on the LEFT parietal region",
    LandmarkName.EURYON_RIGHT: "most lateral point on the RIGHT parietal region",
    LandmarkName.FRONTOTEMPORALE_LEFT: "LEFT frontotemporal junction (anterior, lateral)",
    LandmarkName.FRONTOTEMPORALE_RIGHT: "RIGHT frontotemporal junction (anterior, lateral)",
    LandmarkName.PARIETAL_EMINENCE_LEFT: "most prominent LEFT parietal point",
    LandmarkName.PARIETAL_EMINENCE_RIGHT: "most prominent RIGHT parietal point",
}


# Plausible infant-head longest-bbox extent in mm. Typical infant head
# circumference 41-48 cm => longest linear dimension ~140-200 mm.
INFANT_LONGEST_MM_MIN: float = 140.0
INFANT_LONGEST_MM_MAX: float = 200.0


class ScanMetadata(BaseModel):
    """Provenance for a loaded scan. Emitted to the structured log by
    io.load_scan and carried alongside cleaned meshes for the audit trail
    that arrives in week 3."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    source_format: Literal["stl", "ply", "obj"]
    original_units: Literal["mm", "m", "inch"]
    scale_factor_applied: float = Field(description="1.0 (mm), 1000.0 (m), or 25.4 (inch).")
    n_vertices: int = Field(ge=3)
    n_faces: int = Field(ge=1)
