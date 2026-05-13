"""Parametric synthetic head primitive.

This is the stand-in for a real patient scan until week 2. The output is a
watertight `trimesh.Trimesh` in the canonical reference frame:

    Origin: between the ear canals at skull base.
    +X: patient's right.
    +Y: anterior.
    +Z: superior.

Approach:

1. Start from a unit icosphere (deterministic vertex/face ordering since the
   subdivisions are fixed).
2. Apply per-vertex anisotropic scaling to get base ellipsoid axes
   (X = width/2, Y = length/2, Z = height/2).
3. Lift the head so its base sits at Z=0 (we want the origin near the ear
   canals at skull base, so we put the bottom of the ellipsoid at Z=0 and
   rely on the centroid being inside the head).
4. Apply deformations as vertex displacements along the outward normal:
   - Occipital flattening: Gaussian indent on the posterior (-Y) side.
   - Frontal bossing:      Gaussian protrusion on the anterior (+Y) side.
   - Brachycephaly: scale the Y coordinate (post-deformation) by the
     compression factor.
"""

from __future__ import annotations

import numpy as np
import trimesh

from kernel.config import HeadConfig


def _occipital_center(head: HeadConfig) -> np.ndarray:
    """Compute the center of the occipital-flattening Gaussian.

    Posterior is -Y. We place the center on the posterior surface of the
    base ellipsoid at z = 0.5 * height_mm (mid-superior on the back of the
    head), shifted laterally by the asymmetry parameter.
    """
    return np.array(
        [
            head.occipital_flat_lateral_offset_mm,
            -head.length_mm / 2.0,
            head.height_mm * 0.5,
        ],
        dtype=np.float64,
    )


def _frontal_center(head: HeadConfig) -> np.ndarray:
    """Center of the frontal-bossing Gaussian — anterior (+Y), mid-superior."""
    return np.array(
        [0.0, head.length_mm / 2.0, head.height_mm * 0.55],
        dtype=np.float64,
    )


def _gaussian_displacement(
    vertices: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    magnitude_mm: float,
    radius_mm: float,
    direction_sign: float,
) -> np.ndarray:
    """Return per-vertex displacement vectors for one Gaussian deformation.

    direction_sign: +1 means push outward along the normal (protrusion),
                    -1 means push inward (indent).
    """
    if magnitude_mm <= 0:
        return np.zeros_like(vertices)
    # Distance from each vertex to the deformation center.
    d = np.linalg.norm(vertices - center, axis=1)
    # Standard Gaussian: e^(-d^2 / (2 sigma^2)). We use radius_mm as the
    # 1-sigma falloff so the displacement is ~60% at one radius_mm out
    # and ~0 at 3 sigma.
    weight = np.exp(-(d**2) / (2.0 * radius_mm**2))
    scaled = magnitude_mm * direction_sign * weight
    return normals * scaled[:, None]


def generate_test_head(config: HeadConfig) -> trimesh.Trimesh:
    """Generate the parametric synthetic head mesh.

    Returns a watertight, manifold trimesh.Trimesh in the canonical frame.
    """
    # 1. Base icosphere. trimesh's icosphere has deterministic ordering.
    sphere = trimesh.creation.icosphere(subdivisions=config.subdivisions, radius=1.0)
    verts = np.array(sphere.vertices, dtype=np.float64)
    faces = np.array(sphere.faces, dtype=np.int64)

    # 2. Anisotropic scaling to base ellipsoid.
    scale = np.array(
        [config.width_mm / 2.0, config.length_mm / 2.0, config.height_mm / 2.0],
        dtype=np.float64,
    )
    verts = verts * scale

    # 3. Lift so the bottom of the ellipsoid is near z = 0. The origin sits
    # roughly at the ear canals, which we model as the level where the
    # ellipsoid crosses z = 0 from below; we leave the ellipsoid centered
    # on z = 0 and rely on the trim line cutting below the ear canals.
    # No lift needed — the canonical frame has +Z = superior and the
    # ellipsoid is symmetric in Z, so the head extends from -height/2 to
    # +height/2. The trim happens below later.
    # (See CLAUDE.md reference frame note.)

    # 4. Compute outward normals at the current ellipsoid surface. For a
    # scaled icosphere, the outward direction at vertex v is roughly v
    # itself (it's centered on origin), but the *true* outward normal on an
    # ellipsoid is the gradient of the implicit function and is NOT v/|v|.
    # Use the gradient: normal_unnormalized = (vx/a^2, vy/b^2, vz/c^2).
    a, b, c = scale
    grad = np.column_stack([verts[:, 0] / (a * a), verts[:, 1] / (b * b), verts[:, 2] / (c * c)])
    norm_len = np.linalg.norm(grad, axis=1, keepdims=True)
    # Avoid division by zero at any pathological vertex.
    norm_len = np.where(norm_len < 1e-12, 1.0, norm_len)
    normals = grad / norm_len

    # 5. Apply Gaussian deformations as displacements along the normals.
    disp = np.zeros_like(verts)
    disp += _gaussian_displacement(
        verts,
        normals,
        _occipital_center(config),
        config.occipital_flat_mm,
        config.occipital_flat_radius_mm,
        direction_sign=-1.0,
    )
    disp += _gaussian_displacement(
        verts,
        normals,
        _frontal_center(config),
        config.frontal_bossing_mm,
        config.frontal_bossing_radius_mm,
        direction_sign=+1.0,
    )
    verts = verts + disp

    # 6. Brachycephaly: AP compression coefficient applied along Y.
    if config.brachycephaly_factor != 1.0:
        verts[:, 1] = verts[:, 1] * config.brachycephaly_factor

    # 7. Build the trimesh. Faces are unchanged from the base icosphere, so
    # topology stays manifold and watertight as long as we did not invert
    # any triangles (we did not).
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # process=False keeps vertex/face ordering bit-identical across runs.
    # We do explicitly fix winding to make sure normals point outward
    # post-deformation. fix_normals is deterministic given a fixed mesh.
    mesh.fix_normals()

    return mesh
