"""Headless tests for the annotation tool.

The interactive PyVista session (run_annotation_tool) needs a display and is
out of scope for unit tests — it is marked `# pragma: no cover` and verified
manually (acceptance #2). Everything testable lives in AnnotationState and
annotate_programmatic, exercised here with no display.
"""

from __future__ import annotations

import numpy as np
import pytest

from anatomy.annotate import AnnotationState, annotate_programmatic
from anatomy.config import LANDMARK_ORDER, REGISTRATION_LANDMARKS, LandmarkName
from anatomy.errors import AnatomyError
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.synthetic import derive_landmarks_from_synthetic
from kernel.config import HeadConfig
from kernel.primitives.test_head import generate_test_head


def _lm(name: LandmarkName) -> Landmark:
    return Landmark(name=name, position_mm=(1.0, 2.0, 3.0), method="manual")


def test_state_cursor_advances_and_completes() -> None:
    s = AnnotationState()
    assert s.current == LANDMARK_ORDER[0]
    assert s.remaining() == 12
    for name in LANDMARK_ORDER:
        assert s.current == name
        s.record(_lm(name))
    assert s.done
    assert s.remaining() == 0


def test_state_skip_back_restart() -> None:
    s = AnnotationState()
    s.record(_lm(LANDMARK_ORDER[0]))
    s.skip()  # skip LANDMARK_ORDER[1]
    assert s.current == LANDMARK_ORDER[2]
    s.back()  # back to LANDMARK_ORDER[1]
    assert s.current == LANDMARK_ORDER[1]
    s.back()  # back to [0], drops its pick
    assert s.current == LANDMARK_ORDER[0]
    assert LANDMARK_ORDER[0] not in s.picks
    s.record(_lm(LANDMARK_ORDER[0]))
    s.restart()
    assert s.current == LANDMARK_ORDER[0]
    assert s.picks == {}


def test_state_missing_required_blocks_save() -> None:
    s = AnnotationState()
    # Pick only non-registration landmarks.
    for name in LANDMARK_ORDER:
        if name in REGISTRATION_LANDMARKS:
            s.skip()
        else:
            s.record(_lm(name))
    assert set(s.missing_required()) == set(REGISTRATION_LANDMARKS)
    with pytest.raises(AnatomyError, match="missing required"):
        s.to_landmark_set()


def test_status_text_mentions_shortcuts_and_progress() -> None:
    s = AnnotationState()
    txt = s.status_text()
    assert "GLABELLA" in txt
    assert "12 landmark(s) remaining" in txt
    assert "[n]" in txt and "[b]" in txt and "[s]" in txt


def test_annotate_programmatic_snaps_all_twelve() -> None:
    cfg = HeadConfig()
    mesh = generate_test_head(cfg)
    truth = derive_landmarks_from_synthetic(cfg)
    # Feed the ground-truth positions back in as if a human clicked them
    # (offset slightly off-surface so the snap actually does something).
    picks = []
    for name in LANDMARK_ORDER:
        p = truth.landmarks[name].as_array() + np.array([0.0, 0.0, 2.0])
        picks.append((name, p))
    result = annotate_programmatic(mesh, picks)
    assert isinstance(result, LandmarkSet)
    assert result.frame == "source"
    assert set(result.landmarks) == set(LANDMARK_ORDER)
    assert all(lm.method == "manual" for lm in result.landmarks.values())
    # Snapped points must lie on the mesh surface.
    import trimesh

    pts = np.array([result.landmarks[n].position_mm for n in LANDMARK_ORDER])
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    assert float(np.max(dist)) < 1e-3


def test_annotate_programmatic_partial_raises_on_missing_required() -> None:
    cfg = HeadConfig()
    mesh = generate_test_head(cfg)
    truth = derive_landmarks_from_synthetic(cfg)
    # Provide only the non-registration landmarks -> required ones missing.
    picks = [
        (n, truth.landmarks[n].as_array())
        for n in LANDMARK_ORDER
        if n not in REGISTRATION_LANDMARKS
    ]
    with pytest.raises(AnatomyError, match="missing required"):
        annotate_programmatic(mesh, picks)
