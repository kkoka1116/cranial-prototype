"""Pydantic v2 configuration models for the kernel.

These types are the kernel's input contract. Every kernel function takes one
of these (or a field of one of these) — never loose kwargs. When the AI layer
arrives in week 5+, it will produce instances of these types and the kernel
will execute them unchanged.

Units: every numeric field is millimeters unless its name carries another
unit suffix (e.g. `density_g_per_cm3`, `mass_g`).

Reference frame (every mesh, every point):
    Origin: between the ear canals, at skull base.
    +X: patient's right (lateral).
    +Y: anterior.
    +Z: superior.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Synthetic head (stand-in for a real scan until week 2)
# ---------------------------------------------------------------------------


class HeadConfig(BaseModel):
    """Parameters for the synthetic-head primitive.

    A base ellipsoid with axes (length_mm, width_mm, height_mm) at typical
    8-month-old infant dimensions, plus three controllable deformations:

    - Occipital flattening (Gaussian indent on the posterior, with an
      optional lateral offset for asymmetric / plagiocephalic shapes).
    - Frontal bossing (Gaussian protrusion on the anterior).
    - Brachycephaly factor (AP compression coefficient < 1.0 compresses
      the head front-to-back; > 1.0 elongates it).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Base ellipsoid (defaults: typical 8-month-old infant)
    length_mm: float = Field(default=165.0, gt=0, description="AP axis (Y).")
    width_mm: float = Field(default=130.0, gt=0, description="ML axis (X).")
    height_mm: float = Field(default=115.0, gt=0, description="SI axis (Z).")

    # Resolution: trimesh icosphere subdivisions. 5 = ~10k faces, plenty for
    # week 1 visual + thickness checks while staying fast.
    subdivisions: int = Field(default=5, ge=2, le=7)

    # Occipital flattening (Gaussian indent on posterior, -Y direction)
    occipital_flat_mm: float = Field(
        default=0.0, ge=0, description="Indent depth at center of flat region."
    )
    occipital_flat_radius_mm: float = Field(
        default=45.0, gt=0, description="Gaussian falloff radius."
    )
    occipital_flat_lateral_offset_mm: float = Field(
        default=0.0,
        description=(
            "Lateral offset of the indent center along +X. "
            "Positive offset = right-side plagiocephaly."
        ),
    )

    # Frontal bossing (Gaussian protrusion on anterior, +Y direction)
    frontal_bossing_mm: float = Field(default=0.0, ge=0)
    frontal_bossing_radius_mm: float = Field(default=40.0, gt=0)

    # Brachycephaly: AP compression coefficient applied to the Y axis.
    # 1.0 = no compression. 0.85 = compressed (brachycephalic).
    brachycephaly_factor: float = Field(default=1.0, gt=0, le=2.0)


# ---------------------------------------------------------------------------
# Per-region corrections — the inward dents on the helmet outer surface
# that apply pressure to remodel asymmetry.
# ---------------------------------------------------------------------------


class CorrectionRegion(BaseModel):
    """A single radial-basis correction zone on the helmet outer surface.

    Week 1: expressed as a raw (x, y, z, radius, magnitude) tuple. In week 2
    these will be anchored to anatomical landmarks; in week 4 they get wrapped
    in semantic-object types. The numeric contract here forward-ports cleanly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_mm: float
    y_mm: float
    z_mm: float
    radius_mm: float = Field(gt=0)
    magnitude_mm: float = Field(gt=0, description="Inward correction depth at center, in mm.")
    falloff: str = Field(default="gaussian", pattern="^(gaussian|cosine|linear)$")


# ---------------------------------------------------------------------------
# Trim line — the lower boundary of the helmet.
# ---------------------------------------------------------------------------


class TrimControlPoint(BaseModel):
    """One control point on the helmet trim line.

    Specified in spherical coordinates relative to the head's bounding-sphere
    centroid:
        azimuth_deg: 0 = +Y (anterior), +90 = +X (right), 180 = -Y (posterior).
        elevation_deg: 0 = equator, +90 = +Z (top). For a trim line we expect
                       this to be negative-ish (below the equator).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    azimuth_deg: float = Field(ge=-360.0, le=360.0)
    elevation_deg: float = Field(ge=-90.0, le=90.0)


class TrimConfig(BaseModel):
    """Trim-line configuration. Must form a closed loop around the head."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_points: list[TrimControlPoint] = Field(min_length=4)
    # How far below the lowest control point on +Z to extend the cutting body.
    # Bigger = safer cut; doesn't affect the resulting helmet shape.
    cut_depth_mm: float = Field(default=80.0, gt=0)

    @model_validator(mode="after")
    def _check_unique_azimuths(self) -> TrimConfig:
        azimuths = [cp.azimuth_deg % 360.0 for cp in self.control_points]
        if len(set(azimuths)) != len(azimuths):
            raise ValueError("trim control points must have distinct azimuths (modulo 360)")
        return self


# ---------------------------------------------------------------------------
# Shell — the outer/inner offset construction.
# ---------------------------------------------------------------------------


class ShellConfig(BaseModel):
    """Helmet shell construction parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relief_mm: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Uniform gap between head and inner shell surface " "(the inner offset distance)."
        ),
    )
    wall_thickness_mm: float = Field(
        default=4.0,
        gt=0,
        description="Nominal shell wall thickness.",
    )
    correction_regions: list[CorrectionRegion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation thresholds.
# ---------------------------------------------------------------------------


class ValidationConfig(BaseModel):
    """Thresholds for the validation suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_wall_thickness_mm: float = Field(default=3.0, gt=0)
    max_mass_g: float = Field(default=250.0, gt=0)
    max_bbox_extent_mm: float = Field(default=250.0, gt=0)
    # PETG density. Brief specifies 1.27 g/cm^3.
    material_density_g_per_cm3: float = Field(default=1.27, gt=0)


# ---------------------------------------------------------------------------
# Top-level kernel input.
# ---------------------------------------------------------------------------


class KernelConfig(BaseModel):
    """Top-level kernel input — everything needed to generate one helmet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Short config identifier.")
    head: HeadConfig = Field(default_factory=HeadConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    trim: TrimConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> KernelConfig:
        """Load and validate a KernelConfig from a YAML file."""
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"YAML at {path} must be a mapping")
        return cls.model_validate(data)
