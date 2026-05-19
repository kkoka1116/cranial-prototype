"""The thesis-proving test (the week's final gate).

One hand-authored Design, run against two different patients (baseline vs
right-plagiocephaly synthetic heads). If anchoring is functional rather
than cosmetic, the SAME semantic anchor must resolve to each patient's
ACTUAL occipital geometry, so the two helmets differ — and differ in the
anatomically expected place (the posterior occiput, where the two heads'
shapes diverge). If this cannot be shown, the synthesis layer does not
actually work.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh

from synthesis.pipeline import synthesize
from synthesis.resolve import resolve_anchors


def _helmet(repo, h) -> trimesh.Trimesh:
    return trimesh.load(file_obj=io.BytesIO(repo.load_blob(h)), file_type="stl", process=False)


def test_same_design_two_patients_diverge_anatomically(
    make_case_env, baseline_cfg, plagio_cfg, design_factory
):
    repo_b, scan_b = make_case_env(baseline_cfg, "baseline")
    repo_p, scan_p = make_case_env(plagio_cfg, "plagio")
    design = design_factory()  # the SAME design for both patients
    correction = design.semantic_objects[0]  # the CorrectionZone

    # 1. The same semantic anchor resolves to DIFFERENT patient-specific
    #    coordinates — the occiput sits in a different place on a
    #    plagiocephalic head than on a normocephalic one.
    base_mesh = _helmet(repo_b, scan_b.mesh_hash)
    plag_mesh = _helmet(repo_p, scan_p.mesh_hash)
    c_base = np.array(resolve_anchors(correction, scan_b.landmark_set, base_mesh).center_xyz)
    c_plag = np.array(resolve_anchors(correction, scan_p.landmark_set, plag_mesh).center_xyz)
    anchor_shift = float(np.linalg.norm(c_base - c_plag))
    assert anchor_shift > 3.0, (
        f"the correction anchor resolved to nearly the same point on two "
        f"different patients ({anchor_shift:.2f} mm) — anchoring is cosmetic"
    )

    # 2. Same design -> two different helmets (not a fixed template).
    r_base = synthesize(design, scan_b, repo_b)
    r_plag = synthesize(design, scan_p, repo_p)
    assert r_base.helmet_hash != r_plag.helmet_hash

    # 3. The helmets differ in the anatomically expected PLACE: the largest
    #    geometric divergence is posterior (-Y, the occiput) — where the
    #    plagiocephalic flattening actually is — not smeared arbitrarily.
    hb = _helmet(repo_b, r_base.helmet_hash)
    hp = _helmet(repo_p, r_plag.helmet_hash)
    samples, _ = trimesh.sample.sample_surface_even(hp, 6000, seed=0)
    _, dist, _ = trimesh.proximity.closest_point(hb, samples)
    worst = samples[int(np.argmax(dist))]
    assert float(np.max(dist)) > 2.0, "helmets barely differ — anchoring not functional"
    assert worst[1] < 0.0, (
        f"largest divergence is not posterior (y={worst[1]:.1f}) — the "
        f"patient-specific difference is not landing on the occiput"
    )


def test_patient_specific_correction_tracks_each_occiput(
    make_case_env, baseline_cfg, plagio_cfg, design_factory
):
    """The resolved correction center on each patient lies on (snapped to)
    that patient's own posterior surface, not a shared template point."""
    repo_b, scan_b = make_case_env(baseline_cfg, "baseline")
    repo_p, scan_p = make_case_env(plagio_cfg, "plagio")
    correction = design_factory().semantic_objects[0]

    for repo, scan in ((repo_b, scan_b), (repo_p, scan_p)):
        head = _helmet(repo, scan.mesh_hash)
        center = np.array(resolve_anchors(correction, scan.landmark_set, head).center_xyz)
        # Centroid of {opisthocranion, euryon_right}: posterior-ish, and
        # within a plausible distance of this patient's head surface.
        _, d, _ = trimesh.proximity.closest_point(head, center.reshape(1, 3))
        assert center[1] < 0.0, "correction center should be posterior (-Y)"
        assert float(d[0]) < 60.0, "correction center is implausibly far from the head"
